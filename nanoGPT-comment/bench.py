"""
nanoGPT 基准测试脚本

用于性能测试和模型基准评估的简化版训练脚本。
支持两种模式：
1. 简单基准测试：测量训练迭代时间和 MFU
2. PyTorch Profiler：使用 tensorboard 可视化性能分析

使用方式：
$ python bench.py
$ python bench.py --profile=True
"""

import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# =============================================================================
# 配置参数
# =============================================================================

batch_size = 12          # 批量大小
block_size = 1024         # 上下文长度
bias = False              # 是否使用偏置
real_data = True          # 是否使用真实数据（否则使用随机数据）
seed = 1337               # 随机种子
device = 'cuda'           # 设备：'cpu', 'cuda', 'cuda:0', 'mps' 等

# 数据类型：优先 bfloat16
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# 是否使用 PyTorch 2.0 编译加速
compile = True

# 是否使用 PyTorch Profiler（会生成 tensorboard 日志）
profile = False

# =============================================================================
# 配置覆盖
# =============================================================================
exec(open('configurator.py').read())

# =============================================================================
# 初始化
# =============================================================================

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 允许 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# =============================================================================
# 数据加载
# =============================================================================

if real_data:
    # 使用真实数据集（OpenWebText）
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        """获取一批训练数据（split 参数在本脚本中被忽略）"""
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

else:
    # 使用随机数据（用于纯粹的性能测试，不关心数据加载）
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)


# =============================================================================
# 模型初始化
# =============================================================================

gptconf = GPTConfig(
    block_size=block_size,  # 上下文长度
    n_layer=12,            # Transformer 层数
    n_head=12,             # 注意力头数
    n_embd=768,            # 嵌入维度
    dropout=0,              # Dropout（设为 0 以保证确定性）
    bias=bias,
)
model = GPT(gptconf)
model.to(device)

# 配置优化器
optimizer = model.configure_optimizers(
    weight_decay=1e-2,
    learning_rate=1e-4,
    betas=(0.9, 0.95),
    device_type=device_type
)

# 编译模型（需要 PyTorch 2.0）
if compile:
    print("正在编译模型...")
    model = torch.compile(model)


# =============================================================================
# 性能测试
# =============================================================================

if profile:
    # PyTorch Profiler 模式
    # 参考资料：
    # - 教程：https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - API：https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile

    wait, warmup, active = 5, 5, 5  # 预热、活跃阶段步数
    num_steps = wait + warmup + active

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,   # 会有额外开销，不需要时可禁用
        with_flops=True,
        with_modules=False,  # 目前只用于 torchscript 模型
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            # 通知 profiler 当前步骤完成
            prof.step()

else:
    # 简单基准测试模式
    torch.cuda.synchronize()

    for stage, num_steps in enumerate([10, 20]):
        # stage 0: burn-in（预热）
        # stage 1: benchmark（正式测试）

        t0 = time.time()
        X, Y = get_batch('train')

        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # 估算 MFU（Model FLOPs Utilization）
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)

        if stage == 1:
            print(f"每次迭代耗时: {dt / num_steps * 1000:.4f}ms, MFU: {mfu * 100:.2f}%")