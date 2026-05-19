"""
train_gpt2.py - GPT-2 训练脚本
====================================

本脚本实现了从零训练 GPT-2 模型的核心功能，支持：
- 分布式数据并行 (DDP) 训练
- 混合精度训练 (bfloat16)
- 梯度累积
- 学习率调度（warmup + cosine decay）
- HellaSwag 评估
- 文本生成采样

主要组件：
1. GPT 模型定义（CausalSelfAttention、MLP、Block）
2. 数据加载器（DataLoaderLite）
3. 分布式训练设置
4. 训练循环（包含验证、评估、生成）

使用方式：
    # 单 GPU 训练
    python train_gpt2.py

    # 多 GPU 分布式训练（8卡）
    torchrun --standalone --nproc_per_node=8 train_gpt2.py
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# =============================================================================
# 第一部分：GPT 模型定义
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层 (Causal Self-Attention)

    这是 Transformer 的核心组件，实现了自注意力机制：

    1. 将输入 x 通过三个线性投影得到 Query (Q)、Key (K)、Value (V)
    2. 计算 Q 和 K 的点积，得到注意力分数
    3. 使用 softmax 归一化得到注意力权重
    4. 用注意力权重对 V 加权求和，得到输出

    "因果" (Causal) 意味着：当前位置只能看到之前位置的信息，
    通过下三角掩码（is_causal=True）实现，确保自回归生成的可行性。

    参数：
        config: GPTConfig 配置对象，包含 n_embd（嵌入维度）和 n_head（头数）

    前向传播：
        输入: x shape = (B, T, C)  其中 B=batch_size, T=序列长度, C=嵌入维度
        输出: y shape = (B, T, C)
    """

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度能被头数整除
        assert config.n_embd % config.n_head == 0

        # QKV 投影：将嵌入向量线性变换为 3 倍宽度的向量（包含 Q、K、V）
        # 输入: (B, T, n_embd) -> 输出: (B, T, 3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影：将多头注意力的结果投影回原始维度
        # 输入: (B, T, n_embd) -> 输出: (B, T, n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # NANOGPT_SCALE_INIT 是一个标记，用于在初始化时进行特殊缩放
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # 保存配置参数
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，shape = (B, T, C)

        返回:
            y: 输出张量，shape = (B, T, C)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # 计算 QKV：先将 x 线性变换，再分割为 Q、K、V
        # qkv shape: (B, T, 3 * n_embd)
        # q, k, v shape: 各 (B, T, n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 重塑为多头形式：
        # (B, T, n_embd) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # 其中 head_size = n_embd / n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 计算注意力：使用 PyTorch 的 flash attention（高效实现）
        # is_causal=True 会自动应用因果掩码
        # y shape: (B, n_head, T, head_size)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 重新组装所有头的输出
        # (B, n_head, T, head_size) -> (B, T, n_head * head_size) = (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    MLP 前馈网络 (Multi-Layer Perceptron)

    Transformer 中的前馈网络部分，由两个线性层组成：
    1. 扩展层：将维度从 n_embd 扩展到 4 * n_embd
    2. 激活函数：GELU（高斯误差线性单元）
    3. 收缩层：将维度从 4 * n_embd 收缩回 n_embd

    GELU 激活函数比 ReLU 更平滑，能够提供更好的梯度流。

    参数：
        config: GPTConfig 配置对象

    前向传播：
        输入: x shape = (B, T, n_embd)
        输出: x shape = (B, T, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        # 扩展层：n_embd -> 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU 激活函数（使用 tanh 近似，更快）
        self.gelu = nn.GELU(approximate='tanh')

        # 收缩层：4 * n_embd -> n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # 标记，用于特殊初始化
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """前向传播"""
        x = self.c_fc(x)      # 扩展维度
        x = self.gelu(x)      # 激活
        x = self.c_proj(x)    # 收缩维度
        return x


class Block(nn.Module):
    """
    Transformer 块

    一个完整的 Transformer 块包含：
    1. LayerNorm (ln_1)
    2. CausalSelfAttention (attn)
    3. 残差连接 (x + attn)
    4. LayerNorm (ln_2)
    5. MLP (mlp)
    6. 残差连接 (x + mlp)

    残差连接（Skip Connection）是 Transformer 训练稳定的关键：
    - 缓解梯度消失问题
    - 允许梯度直接反向传播到浅层
    - 使得构建深层网络成为可能

    参数：
        config: GPTConfig 配置对象

    前向传播：
        输入: x shape = (B, T, n_embd)
        输出: x shape = (B, T, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        # 第一个 LayerNorm + 自注意力
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        # 第二个 LayerNorm + MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """前向传播：注意力 + 残差 -> MLP + 残差"""
        # 自注意力 + 残差连接
        x = x + self.attn(self.ln_1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    GPT 模型配置数据类

    属性：
        block_size: 最大序列长度（位置编码大小），GPT-2 为 1024
        vocab_size: 词汇表大小，GPT-2 为 50257
        n_layer: Transformer 层的数量
        n_head: 注意力头的数量
        n_embd: 嵌入维度（也称为 hidden size）
    """
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):
    """
    GPT 模型主体

    完整的 GPT 模型包含：
    1. Token 嵌入层 (wte): 将 token ID 转换为嵌入向量
    2. 位置嵌入层 (wpe): 为每个位置添加位置信息
    3. Transformer 块堆叠 (h): n_layer 个 Block
    4. 最终 LayerNorm (ln_f)
    5. LM Head (lm_head): 将嵌入映射回词汇表空间

    权重共享：token 嵌入和 LM head 共享权重，这是 GPT 的常用做法，
    可以减少参数量并提高训练效果。

    参数：
        config: GPTConfig 配置对象

    前向传播：
        输入:
            idx: token IDs, shape = (B, T)
            targets: 目标 token IDs, shape = (B, T)，可选
        输出:
            logits: shape = (B, T, vocab_size)
            loss: 交叉熵损失，仅当 targets 不为 None 时返回
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 创建 Transformer 组件
        self.transformer = nn.ModuleDict(dict(
            # Token 嵌入：vocab_size 个 token，每个 token 对应 n_embd 维向量
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入：block_size 个位置，每个位置对应 n_embd 维向量
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Transformer 块堆叠
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终 LayerNorm
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # LM Head：将隐藏状态映射到词汇表空间，得到每个位置的 logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：token 嵌入和 LM head 使用相同的权重
        # 这意味着预测某个 token 时，"看到"的是同一个嵌入向量
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化模型参数

        初始化策略：
        - 线性层：使用正态分布初始化，标准差为 0.02
        - 对于带有 NANOGPT_SCALE_INIT 标记的层（如 attention output 和 MLP output），
          标准差需要乘以 (2 * n_layer)^(-0.5)，这是为了保持残差路径上方差一致
        - 偏置：初始化为零
        - 嵌入层：使用正态分布初始化，标准差为 0.02
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 缩放初始化：std *= (2 * n_layer)^(-0.5)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播

        参数:
            idx: token IDs, shape = (B, T)
            targets: 目标 token IDs, shape = (B, T)，可选

        返回:
            logits: shape = (B, T, vocab_size)
            loss: 交叉熵损失，仅当 targets 不为 None 时返回
        """
        # idx shape: (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # 创建位置索引：[0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)

        # 位置嵌入：shape = (T, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # Token 嵌入：shape = (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 合并 token 嵌入和位置嵌入
        x = tok_emb + pos_emb

        # 通过所有 Transformer 块
        for block in self.transformer.h:
            x = block(x)

        # 最终 LayerNorm
        x = self.transformer.ln_f(x)

        # LM Head：计算 logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果提供了 targets）
        loss = None
        if targets is not None:
            # 交叉熵损失：展平为 (B*T, vocab_size) 和 (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        从 Hugging Face 加载预训练的 GPT-2 模型权重

        参数:
            model_type: 模型类型，可以是 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        返回:
            model: 加载了预训练权重的 GPT 模型
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # 根据模型类型设置配置参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # GPT model checkpoints 总是 50257
        config_args['block_size'] = 1024   # GPT model checkpoints 总是 1024

        # 创建从头初始化的 GPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # 过滤掉 attn.bias 这个 buffer（不是参数）
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 加载 Hugging Face 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 对齐参数并复制
        sd_keys_hf = sd_hf.keys()
        # 忽略这些 buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # 需要转置的权重（因为 Hugging Face 使用 Conv1D）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # 确保参数数量一致
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Conv1D 权重需要转置
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通参数直接复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        配置优化器

        将模型参数分为两组：
        1. 需要权重衰减的参数（dim >= 2）：主要是线性层的权重矩阵
        2. 不需要权重衰减的参数（dim < 2）：主要是偏置、LayerNorm 参数、嵌入权重

        参数:
            weight_decay: 权重衰减系数，通常为 0.1
            learning_rate: 学习率
            device_type: 设备类型，'cuda' 或 'cpu'

        返回:
            optimizer: 配置好的 AdamW 优化器
        """
        # 获取所有需要梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 创建优化组
        # 需要权重衰减：维度 >= 2 的参数（主要是线性层权重）
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # 不需要权重衰减：维度 < 2 的参数（偏置、LayerNorm、嵌入）
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 打印参数统计
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 创建 AdamW 优化器
        # 如果可用，使用 fused 版本（更快）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer


# =============================================================================
# 第二部分：数据加载
# =============================================================================

import tiktoken
import numpy as np


def load_tokens(filename):
    """
    从 .npy 文件加载 token 序列

    参数:
        filename: .npy 文件路径

    返回:
        tokens: torch.Tensor，shape = (num_tokens,)，dtype = torch.long
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # 确保为 int32 类型
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """
    高效的数据加载器

    支持：
    - 流式加载大型数据集（分片存储）
    - 分布式训练（通过 process_rank 和 num_processes）
    - 梯度累积（通过 micro batch）

    使用方式：
        loader = DataLoaderLite(B=64, T=1024, process_rank=0, num_processes=1, split='train')
        x, y = loader.next_batch()  # x, y shape: (B, T)
    """

    def __init__(self, B, T, process_rank, num_processes, split):
        """
        初始化数据加载器

        参数:
            B: micro batch size
            T: sequence length (block_size)
            process_rank: 当前进程的排名（用于分布式训练）
            num_processes: 总进程数（用于分布式训练）
            split: 'train' 或 'val'
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # 获取数据分片文件列表
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        """重置数据加载器状态到初始分片"""
        # 状态：当前分片索引和位置
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # 在分布式训练中，不同进程从不同位置开始读取
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        获取下一个 batch

        返回:
            x: 输入 token IDs, shape = (B, T)
            y: 目标 token IDs, shape = (B, T)
        """
        B, T = self.B, self.T

        # 获取一个 batch 的数据
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # 输入：除了最后一个 token
        y = (buf[1:]).view(B, T)   # 目标：除了第一个 token

        # 前进到下一个 batch 的位置
        self.current_position += B * T * self.num_processes

        # 如果读取下一个 batch 会超出当前分片边界，则切换到下一个分片
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y


# =============================================================================
# 第三部分：辅助函数
# =============================================================================

def get_most_likely_row(tokens, mask, logits):
    """
    在 HellaSwag 评估中，找出可能性最高的结尾

    HellaSwag 是一个多项选择题，每个问题提供 4 个候选结尾。
    这个函数计算每个候选结尾的平均损失，返回损失最低的索引。

    参数:
        tokens: token IDs, shape = (4, max_len) — 4 个候选的 token 序列
        mask: 掩码, shape = (4, max_len) — 1 表示需要评估的位置
        logits: 模型输出, shape = (4, max_len, vocab_size)

    返回:
        pred_norm: 最低损失候选的索引
    """
    # 计算移位后的 logits 和 tokens（预测下一个 token）
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()

    # 展平
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    # 计算每个位置的损失
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # 只对候选结尾部分计算损失（mask == 1 的位置）
    shift_mask = (mask[..., 1:]).contiguous()  # 掩码也需要移位
    masked_shift_losses = shift_losses * shift_mask

    # 求和并平均
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # 返回损失最低的候选索引
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# =============================================================================
# 第四部分：训练设置
# =============================================================================

# 简单的启动方式：
# python train_gpt2.py
# DDP 启动方式（8 GPU）：
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 导入分布式训练相关模块
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# -----------------------------------------------------------------------------
# 设置 DDP（分布式数据并行）
# torchrun 命令会设置 RANK, LOCAL_RANK, WORLD_SIZE 环境变量
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1  # 是否使用 DDP

if ddp:
    # DDP 模式：使用 CUDA 和 NCCL 后端
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

    # 只有主进程（rank=0）进行日志输出和模型保存
    master_process = ddp_rank == 0
else:
    # 非 DDP 模式（单卡或 CPU）
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # 自动检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# 设备类型（用于混合精度训练）
device_type = "cuda" if device.startswith("cuda") else "cpu"

# 设置随机种子
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# 初始化 GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# =============================================================================
# 第五部分：训练超参数
# =============================================================================

# 总 batch size（以 token 数量计）
# 2^19 = 524288 ≈ 0.5M tokens
total_batch_size = 524288

# micro batch size：每个 GPU 每次处理的样本数
B = 64

# 序列长度
T = 1024

# 确保 total_batch_size 可以被整除
assert total_batch_size % (B * T * ddp_world_size) == 0, \
    "make sure total_batch_size is divisible by B * T * ddp_world_size"

# 梯度累积步数
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# 创建数据加载器
train_loader = DataLoaderLite(
    B=B, T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val"
)

# 设置矩阵乘法精度
torch.set_float32_matmul_precision('high')

# =============================================================================
# 第六部分：创建模型
# =============================================================================

# 创建模型（vocab_size 使用 50304 而不是 50257，这是 GPT-2 的一个变体）
model = GPT(GPTConfig(vocab_size=50304))

# 或者从 OpenAI 加载预训练权重：
# model = GPT.from_pretrained("gpt2")

model.to(device)

# torch.compile 会干扰 HellaSwag 评估和生成，暂时禁用
use_compile = False
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 始终获取原始（未包装的）模型
raw_model = model.module if ddp else model

# =============================================================================
# 第七部分：学习率调度
# =============================================================================

# 最大学习率
max_lr = 6e-4

# 最小学习率
min_lr = max_lr * 0.1

# warmup 步数
warmup_steps = 715

# 最大训练步数（约 1 epoch）
max_steps = 19073


def get_lr(it):
    """
    计算当前步的学习率

    学习率调度策略：
    1. 线性 warmup：在 warmup_steps 步内从 0 增加到 max_lr
    2. Cosine decay：从 max_lr 衰减到 min_lr

    参数:
        it: 当前步数

    返回:
        lr: 当前步的学习率
    """
    # 1. 线性 warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2. 如果超过最大步数，返回最小学习率
    if it > max_steps:
        return min_lr

    # 3. Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # coeff 从 1 逐渐变为 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# 第八部分：优化器配置
# =============================================================================

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    device_type=device_type
)

# 创建日志目录
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

# 清空日志文件
with open(log_file, "w") as f:
    pass

# =============================================================================
# 第九部分：训练循环
# =============================================================================

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # -------------------------------------------------------------------------
    # 评估验证损失（每 250 步或最后一步）
    # -------------------------------------------------------------------------
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        # 在分布式训练中聚合所有进程的损失
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # 保存检查点（每 5000 步或最后一步）
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # -------------------------------------------------------------------------
    # 评估 HellaSwag（每 250 步或最后一步）
    # -------------------------------------------------------------------------
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            # 只处理当前进程负责的样本
            if i % ddp_world_size != ddp_rank:
                continue

            # 渲染示例为 tokens
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # 获取 logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)

                pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # 聚合所有进程的统计
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # -------------------------------------------------------------------------
    # 文本生成（每 250 步，除了第 0 步）
    # -------------------------------------------------------------------------
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4  # 生成 4 个样本
        max_length = 32            # 最大生成长度

        # 使用 GPT-2 tokenizer 编码提示
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)

        # 随机数生成器
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        # 自回归生成
        while xgen.size(1) < max_length:
            # 前向传播
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)

                # 只取最后一个位置的 logits
                logits = logits[:, -1, :]  # (B, vocab_size)

                # 转换为概率
                probs = F.softmax(logits, dim=-1)

                # Top-k 采样（k=50）
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                # 从 top-k 中采样
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)

                # 获取对应的 token
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)

                # 追加到序列
                xgen = torch.cat((xgen, xcol), dim=1)

        # 打印生成的文本
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # -------------------------------------------------------------------------
    # 训练步骤
    # -------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # 在 DDP 模式中，只在最后一个 micro_step 同步梯度
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        # 混合精度前向传播
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # 损失需要除以梯度累积步数
        # 因为梯度是累加的，而我们需要的是平均损失
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    # 聚合所有进程的损失
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # 梯度裁剪
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 设置学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    # 等待 GPU 完成
    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0  # 时间差（秒）
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
              f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# 清理分布式进程组
if ddp:
    destroy_process_group()