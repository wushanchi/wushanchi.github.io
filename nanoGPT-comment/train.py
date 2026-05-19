"""
nanoGPT 训练脚本

本训练脚本可以在两种模式下运行：
1. 单 GPU 调试模式
2. 分布式数据并行（DDP）大规模训练

运行示例：
- 单 GPU 运行：
  $ python train.py --batch_size=32 --compile=False

- 单节点 4 GPU DDP 运行：
  $ torchrun --standalone --nproc_per_node=4 train.py

- 多节点 DDP 运行（在第一个主节点上）：
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

- 在工作节点上运行：
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

注意：如果集群没有 Infiniband 互连，请在命令前加上 NCCL_IB_DISABLE=1
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# =============================================================================
# 默认配置：设计用于在 OpenWebText 上训练 GPT-2 (124M)
# =============================================================================

# I/O 配置
out_dir = 'out'                          # 输出目录
eval_interval = 2000                     # 评估间隔（步数）
log_interval = 1                         # 日志打印间隔
eval_iters = 200                         # 每次评估的迭代次数
eval_only = False                        # 若为 True，只在首次评估后退出
always_save_checkpoint = True             # 是否在每次评估后保存检查点
init_from = 'scratch'                    # 初始化方式：'scratch', 'resume', 或 'gpt2*'

# wandb 日志配置
wandb_log = False                         # 默认禁用 wandb 日志
wandb_project = 'owt'                     # wandb 项目名
wandb_run_name = 'gpt2'                   # wandb 运行名称

# 数据配置
dataset = 'openwebtext'                  # 数据集名称
# 梯度累积步数，用于模拟更大的 batch size
gradient_accumulation_steps = 5 * 8
# micro-batch size（如果 gradient_accumulation_steps > 1，这个是每个 micro-batch 的大小）
batch_size = 12
block_size = 1024                         # 上下文长度（最大位置数）

# 模型配置
n_layer = 12                             # Transformer 层数
n_head = 12                              # 注意力头数
n_embd = 768                             # 嵌入维度
dropout = 0.0                            # Dropout 概率（预训练用 0 微调用 0.1+）
bias = False                              # 是否在 LayerNorm 和 Linear 层使用偏置

# AdamW 优化器配置
learning_rate = 6e-4                     # 最大学习率
max_iters = 600000                       # 总训练步数
weight_decay = 1e-1                      # 权重衰减系数
beta1 = 0.9                              # Adam beta1
beta2 = 0.95                             # Adam beta2
grad_clip = 1.0                           # 梯度裁剪阈值，0.0 表示禁用

# 学习率衰减配置
decay_lr = True                          # 是否衰减学习率
warmup_iters = 2000                      # 预热步数
lr_decay_iters = 600000                  # 学习率衰减步数（应约等于 max_iters）
min_lr = 6e-5                            # 最小学习率（约为 learning_rate/10）

# DDP 配置
backend = 'nccl'                         # 分布式后端：'nccl', 'gloo' 等

# 系统配置
device = 'cuda'                          # 设备：'cpu', 'cuda', 'cuda:0', 'cuda:1'，或 Mac 的 'mps'
# 数据类型：优先使用 bfloat16（如果支持），否则使用 float16
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True                           # 是否使用 PyTorch 2.0 编译模型加速

# =============================================================================
# 配置处理
# =============================================================================

# 提取所有配置键（排除下划线开头的内置变量）
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# 从命令行或配置文件覆盖默认配置
exec(open('configurator.py').read())
# 创建配置字典，用于日志记录
config = {k: globals()[k] for k in config_keys}

# =============================================================================
# 初始化、DDP 设置、I/O 设置
# =============================================================================

# 检测是否运行 DDP
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    # 初始化分布式训练
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])                 # 全局进程排名
    ddp_local_rank = int(os.environ['LOCAL_RANK'])    # 本地 GPU 排名
    ddp_world_size = int(os.environ['WORLD_SIZE'])    # 总进程数
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # 只有主进程负责日志、检查点等
    seed_offset = ddp_rank           # 每个进程使用不同的随机种子

    # 调整梯度累积步数（按进程数缩放）
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 非 DDP 模式（单 GPU）
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# 计算每次迭代处理的 token 数量
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"每次迭代处理的 token 数量: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# 设置随机种子
torch.manual_seed(1337 + seed_offset)

# 允许 TF32 格式（加速矩阵乘法）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设备类型（用于 autocast）
device_type = 'cuda' if 'cuda' in device else 'cpu'

# 自动混合精度配置
# float16 会自动使用 GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# =============================================================================
# 数据加载
# =============================================================================

data_dir = os.path.join('data', dataset)


def get_batch(split):
    """
    获取一个批量的训练/验证数据

    使用 numpy memmap 避免内存泄漏。每次重新创建 memmap 对象，
    因为根据 https://stackoverflow.com/questions/45132940/ 这样做可以避免内存泄漏。

    参数:
        split: 'train' 或 'val'

    返回:
        x: 输入序列，形状为 (batch_size, block_size)
        y: 目标序列（向右偏移一位），形状为 (batch_size, block_size)
    """
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # 随机采样起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # 构建输入和目标序列
    # x: data[i:i+block_size]
    # y: data[i+1:i+1+block_size]（向右偏移一位）
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # 使用 pin_memory 允许异步传输到 GPU
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# =============================================================================
# 模型初始化
# =============================================================================

iter_num = 0
best_val_loss = 1e9

# 尝试从数据集获取词表大小
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"从 {meta_path} 找到词表大小: {meta_vocab_size}")

# 模型参数字典
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,  # 稍后确定
    dropout=dropout
)

if init_from == 'scratch':
    # 从零初始化模型
    print("从零初始化新模型")
    if meta_vocab_size is None:
        print("使用默认词表大小 50304（50257 向上取整到 64 的倍数）")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    # 从检查点恢复训练
    print(f"从 {out_dir} 恢复训练")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # 强制这些配置属性一致，否则无法恢复训练
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # 修复可能的前缀问题
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    # 从 OpenAI GPT-2 权重初始化
    print(f"从 OpenAI GPT-2 权重初始化: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)

    # 记录配置参数用于检查点保存
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# 如果需要，裁剪模型的 block size
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# =============================================================================
# 优化器和编译
# =============================================================================

# 梯度缩放器（dtype 为 float16 时启用）
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 配置优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # 释放内存

# 编译模型（需要 PyTorch 2.0）
if compile:
    print("正在编译模型...（需要约 1 分钟）")
    unoptimized_model = model
    model = torch.compile(model)

# 包装 DDP 模型
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# =============================================================================
# 辅助函数
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """
    估算训练集和验证集的损失

    通过多次批处理取平均来获得更准确的损失估计。

    返回:
        dict: {'train': 训练损失, 'val': 验证损失}
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """
    计算当前迭代的学习率

    学习率调度策略：
    1. 线性预热（前 warmup_iters 步）
    2. 余弦衰减（到 min_lr）

    参数:
        it: 当前迭代步数

    返回:
        float: 当前学习率
    """
    # 1. 线性预热阶段
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2. 超过最大迭代次数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr

    # 3. 余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    # coeff 从 1 衰减到 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# =============================================================================
# 训练循环
# =============================================================================

# 获取第一批数据
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0  # 本进程的生命周期内的迭代次数
raw_model = model.module if ddp else model  # 获取原始模型（去除 DDP 包装）
running_mfu = -1.0  # 运行中的 MFU（模型 FLOPS 利用率）

while True:
    # 确定并设置当前迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 定期评估并保存检查点
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # 转换为百分比
            })

        # 保存最佳模型或每个检查点
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"保存检查点到 {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # 仅评估模式
    if iter_num == 0 and eval_only:
        break

    # 前向-反向更新（带梯度累积）
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # DDP 训练：只在最后一个 micro step 同步梯度
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            # 缩放损失以补偿梯度累积
            loss = loss / gradient_accumulation_steps

        # 异步预取下一批数据
        X, Y = get_batch('train')

        # 反向传播（如果使用 float16 则自动缩放梯度）
        scaler.scale(loss).backward()

    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 更新优化器和缩放器
    scaler.step(optimizer)
    scaler.update()

    # 清除梯度
    optimizer.zero_grad(set_to_none=True)

    # 计时和日志
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # 获取损失值（CPU-GPU 同步点）
        # 缩放回近似真实总损失
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5:  # 让训练循环稳定一下
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

# 清理分布式进程组
if ddp:
    destroy_process_group()