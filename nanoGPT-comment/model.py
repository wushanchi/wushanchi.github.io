"""
nanoGPT 模型定义 - GPT 语言模型的完整实现

本文件包含 GPT 语言模型的完整定义，参考了：
1) OpenAI 官方发布的 GPT-2 TensorFlow 实现
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers 的 PyTorch 实现
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization），支持可选的偏置项。

    与 PyTorch 原生 LayerNorm 的区别：原生版本在 bias=False 时仍然会创建偏置参数，
    本实现真正支持不带偏置的归一化层。

    公式：y = gamma * (x - mean) / sqrt(var + eps) + beta
    其中 gamma 和 beta 是可学习参数。
    """

    def __init__(self, ndim, bias):
        super().__init__()
        # 缩放参数 gamma，初始化为全1
        self.weight = nn.Parameter(torch.ones(ndim))
        # 偏置参数 beta，仅在 bias=True 时创建
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播：执行层归一化

        参数:
            input: 输入张量，形状为 (..., ndim)

        返回:
            归一化后的张量，形状与输入相同
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制（Causal Self-Attention）

    特点：
    1. 使用因果掩码（causal mask）确保每个位置只能看到当前位置及其之前的内容
    2. 支持 Flash Attention（PyTorch 2.0+），可显著加速注意力计算
    3. 实现 Multi-Head Attention，所有注意力头并行计算

    注意力公式：Attention(Q,K,V) = softmax(QK^T / √d_k) V
    """

    def __init__(self, config):
        super().__init__()
        # 确保 embedding 维度能被注意力头数整除
        assert config.n_embd % config.n_head == 0

        # QKV 线性变换：将输入映射为 query、key、value 的拼接
        # 输入: (B, T, n_embd) -> 输出: (B, T, 3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 输出投影：将多头注意力的输出映射回原始维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 正则化 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 保存配置参数
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash Attention 支持检测（PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: 使用慢速注意力。Flash Attention 需要 PyTorch >= 2.0")

            # 注册因果掩码：确保注意力只应用于输入序列的左侧
            # 形状: (1, 1, block_size, block_size)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        前向传播：计算因果自注意力

        参数:
            x: 输入张量，形状为 (B, T, C)，其中
               B = batch size（批量大小）
               T = sequence length（序列长度）
               C = embedding dimensionality（嵌入维度 n_embd）

        返回:
            注意力输出，形状为 (B, T, C)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # 计算 QKV：分割成 query、key、value
        # (B, T, 3*n_embd) -> 3 个 (B, T, n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 重塑为多头注意力格式：(B, T, n_head, head_size) 然后转置为 (B, n_head, T, head_size)
        # 这样可以让所有注意力头并行计算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 因果自注意力计算
        if self.flash:
            # 路径1：使用 Flash Attention CUDA 内核（高效）
            # is_causal=True 自动添加因果掩码
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # 路径2：手动实现注意力机制
            # 1. 计算注意力分数：QK^T / √d_k
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # 2. 应用因果掩码：遮挡未来位置的信息
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            # 3. Softmax 归一化
            att = F.softmax(att, dim=-1)

            # 4. Dropout 正则化
            att = self.attn_dropout(att)

            # 5. 加权求和：注意力权重 × V
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 重新拼接所有注意力头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + Dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron）

    也称为前馈神经网络（FFN），是 Transformer 块中的第二个子层。
    包含两个线性层、一个激活函数和一个 dropout。

    GELU 激活函数：类似于 ReLU 但更平滑，性能通常更好。
    """

    def __init__(self, config):
        super().__init__()
        # 扩展维度：n_embd -> 4 * n_embd（GPT-2 论文中的配置）
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # GELU 激活函数：平滑的非线性激活
        self.gelu = nn.GELU()

        # 压缩回原始维度：4 * n_embd -> n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (B, T, n_embd)

        返回:
            输出张量，形状为 (B, T, n_embd)
        """
        x = self.c_fc(x)      # 线性变换 + 扩展维度
        x = self.gelu(x)      # GELU 激活
        x = self.c_proj(x)   # 线性变换 + 压缩维度
        x = self.dropout(x)  # Dropout 正则化
        return x


class Block(nn.Module):
    """
    Transformer 块（Block）

    每个块包含两个子层：
    1. Multi-Head Self-Attention（带残差连接）
    2. MLP（带残差连接）

    架构：x = x + Attn(LN(x)), x = x + MLP(LN(x))
    """

    def __init__(self, config):
        super().__init__()
        # 子层1：LayerNorm + Causal Self-Attention
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)

        # 子层2：LayerNorm + MLP
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        前向传播：应用两个子层（带残差连接）

        参数:
            x: 输入张量，形状为 (B, T, n_embd)

        返回:
            输出张量，形状为 (B, T, n_embd)
        """
        # 注意力子层：x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))

        # MLP 子层：x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class GPTConfig:
    """
    GPT 模型配置数据类

    默认配置对应 GPT-2 (124M 参数)：
    - 12 层 Transformer
    - 12 个注意力头
    - 768 维嵌入
    - 1024 最大上下文长度
    """
    block_size: int = 1024      # 最大上下文长度（位置编码数量）
    # GPT-2 的词表大小是 50257，这里 padding 到 64 的倍数以提高效率
    vocab_size: int = 50304
    n_layer: int = 12          # Transformer 层数
    n_head: int = 12           # 注意力头数
    n_embd: int = 768          # 嵌入维度
    dropout: float = 0.0       # Dropout 概率
    # 偏置项：True 时与 GPT-2 一致，False 时通常更快更好
    bias: bool = True


class GPT(nn.Module):
    """
    GPT（Generative Pre-trained Transformer）模型

    完整的生成式预训练Transformer实现，包含：
    - Token Embeddings（词嵌入）
    - Position Embeddings（位置编码）
    - 多层 Transformer Block
    - Language Model Head（语言模型头）

    支持从 HuggingFace 加载预训练 GPT-2 权重。
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer 主干网络
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding：将 token ID 映射为嵌入向量
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding：编码位置信息（可学习）
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Token Embedding + Position Embedding 后的 Dropout
            drop = nn.Dropout(config.dropout),
            # Transformer 块列表
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终的 LayerNorm
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language Model Head：将嵌入映射为词表大小的 logits
        # 注意：这里使用 weight tying（权重共享），与 transformer.wte 共享权重
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享警告：当使用 torch.compile() 时会产生警告
        # 此问题目前看来是无害的
        self.transformer.wte.weight = self.lm_head.weight  # 权重绑定 https://paperswithcode.com/method/weight-tying

        # 初始化所有权重
        self.apply(self._init_weights)

        # 对残差投影应用特殊的缩放初始化（GPT-2 论文）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 打印模型参数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        获取模型参数数量

        参数:
            non_embedding: 是否排除位置嵌入（默认为 True）

        说明:
            由于使用了权重共享（weight tying），token embedding 实际作为 lm_head 的权重使用，
            因此在计算非 embedding 参数时需要包含这些。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        权重初始化

        规则：
        - Linear 层：权重 ~ N(0, 0.02)，偏置初始化为 0
        - Embedding 层：权重 ~ N(0, 0.02)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播

        参数:
            idx: 输入 token ID，形状为 (B, T)
            targets: 目标 token ID（可选），用于计算损失

        返回:
            logits: 预测 logits，形状为 (B, T, vocab_size)
            loss: 交叉熵损失（仅当提供 targets 时）
        """
        device = idx.device
        b, t = idx.size()

        # 检查序列长度不超过 block_size
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 位置索引：[0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 前向传播 Transformer
        # 1. Token Embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        # 2. Position Embedding
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        # 3. 残差连接 + Dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        # 4. 通过所有 Transformer 块
        for block in self.transformer.h:
            x = block(x)
        # 5. 最终 LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果提供了目标，计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理优化：只计算最后一个位置的 logits
            # 使用 [-1] 保留时间维度
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        模型裁剪：减小 block size

        例如：可能加载了 GPT-2 预训练模型（block size 1024），
        但想用于更小的模型（较小的 block size）。

        参数:
            block_size: 新的 block size，必须小于等于当前值
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # 裁剪位置嵌入
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # 裁剪注意力掩码
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        从 HuggingFace 加载预训练 GPT-2 权重

        参数:
            model_type: 模型类型，可选 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args: 可选的参数覆盖（仅支持 dropout）

        返回:
            加载了预训练权重的 GPT 模型
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        # 只允许覆盖 dropout
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel
        print(f"从预训练模型加载权重: {model_type}")

        # 模型配置字典
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M 参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M 参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M 参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 参数
        }[model_type]

        print("强制设置 vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        # 覆盖 dropout（如果指定）
        if 'dropout' in override_args:
            print(f"覆盖 dropout 率为 {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # 创建模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 过滤掉注意力掩码缓冲区（不是参数）
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 加载 HuggingFace 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 对齐参数
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # 需要转置的权重（因为 HF 使用 Conv1D）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"参数数量不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Conv1D 权重需要转置
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器

        参数:
            weight_decay: 权重衰减系数
            learning_rate: 学习率
            betas: AdamW 的 beta 参数 (beta1, beta2)
            device_type: 设备类型（'cuda' 或 'cpu'）

        说明:
            按照 GPT-2 论文，只对 2D 参数（矩阵）应用权重衰减，
            偏置和 LayerNorm 参数不进行衰减。
        """
        # 收集所有参数及其名称
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # 过滤出需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 需要权重衰减的参数：维度 >= 2（主要是线性层和嵌入层权重）
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # 不需要权重衰减的参数：维度 < 2（偏置、LayerNorm 缩放/偏移）
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"需要权重衰减的参数: {len(decay_params)} 个，共 {num_decay_params:,} 个参数")
        print(f"不需要权重衰减的参数: {len(nodecay_params)} 个，共 {num_nodecay_params:,} 个参数")

        # 创建 AdamW 优化器，优先使用 fused 版本（GPU）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用 fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估算模型 FLOP 利用率（MFU）

        参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311

        参数:
            fwdbwd_per_iter: 每次迭代的前向+反向传播次数
            dt: 每次迭代耗时（秒）

        返回:
            MFU：实际 FLOPS 与 A100 峰值 FLOPS 的比值
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        # 每个 token 的 FLOPs
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # 每次迭代的总 FLOPs
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # A100 GPU bfloat16 峰值 FLOPS：312 TFLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised

        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成

        参数:
            idx: 条件序列，形状为 (B, T) 的 LongTensor
            max_new_tokens: 要生成的新 token 数量
            temperature: 温度参数（控制随机性），1.0 表示不调整
            top_k: 如果指定，只保留概率最高的 k 个 token

        返回:
            生成后的序列，形状为 (B, T + max_new_tokens)

        说明:
            每次生成一个 token，将其追加到序列末尾，继续输入模型。
            这实现了自回归（autoregressive）生成。
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播获取 logits
            logits, _ = self(idx_cond)

            # 只取最后一个位置的 logits（用于预测下一个 token）
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax 转为概率分布
            probs = F.softmax(logits, dim=-1)

            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx