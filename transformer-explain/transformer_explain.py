"""
Transformer Architecture - 完整 PyTorch 实现
结合 "Attention Is All You Need" (Vaswani et al., 2017)

详细解释每个模块的输入、输出维度和工作原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional

# ============================================================
# 配置参数
# ============================================================
class Config:
    """Transformer 配置参数"""
    vocab_size: int = 10000      # 词表大小
    d_model: int = 512           # 模型维度 (embedding & hidden)
    num_heads: int = 8           # 注意力头数
    num_layers: int = 6          # encoder/decoder 层数
    d_ff: int = 2048             # FFN 隐藏层维度 (4 * d_model)
    max_len: int = 5000          # 最大序列长度
    dropout: float = 0.1         # Dropout 概率

cfg = Config()


# ============================================================
# 1. 输入编码 (Input Embedding & Positional Encoding)
# ============================================================

class InputEmbedding(nn.Module):
    """
    输入: [B, L] - B=batch_size, L=seq_len
    输出: [B, L, D] - D=d_model=512
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L] token IDs
        # 输出: [B, L, D] 每个 token 对应一个 d_model 维向量
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    注入位置信息，让 Transformer 知道 token 的顺序
    
    输入: [B, L, D] - 已经 embedding 的序列
    输出: [B, L, D] - 加上位置编码，维度不变
    
    使用 sin/cos 函数:
        PE(pos, 2i)   = sin(pos / 10000^(2i/D))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 频率项 (用对数避免大数值)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1, 3, 5, ...
        
        # 添加 batch 维度: [1, max_len, d_model] 方便 broadcast
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        # 将位置编码加到输入上 (自动截取有效长度)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def demo_input_encoding():
    """演示输入编码的维度变化"""
    print("\n" + "="*60)
    print("1. INPUT ENCODING - 演示")
    print("="*60)
    
    # 模拟输入: batch=2, seq_len=5
    x = torch.randint(0, cfg.vocab_size, (2, 5))
    print(f"输入 Token IDs: {x}")
    print(f"输入 Shape: {x.shape}  # [B, L]")
    print(f"  B={x.size(0)} (batch_size)")
    print(f"  L={x.size(1)} (seq_len)")
    
    # Embedding 层
    embed = InputEmbedding(cfg.vocab_size, cfg.d_model)
    embedded = embed(x)
    print(f"\nEmbedding 后:")
    print(f"  Shape: {embedded.shape}  # [B, L, D]")
    print(f"  D={cfg.d_model}")
    
    # 位置编码
    pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len)
    encoded = pos_enc(embedded)
    print(f"\n加位置编码后:")
    print(f"  Shape: {encoded.shape}  # [B, L, D]")
    print(f"  维度不变，位置信息已注入")
    
    return encoded  # [B, L, D]


# ============================================================
# 2. 多头注意力 (Multi-Head Attention)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    输入: [B, L, D]
    输出: [B, L, D]
    
    核心公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    多头:
        - 将 D=512 维分成 h=8 个头
        - 每头 d_k = D/h = 64
        - 各头独立计算 attention，最后 concat
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每头维度 64
        
        # Q, K, V 的投影矩阵
        self.W_q = nn.Linear(d_model, d_model)  # [D, D]
        self.W_k = nn.Linear(d_model, d_model)  # [D, D]
        self.W_v = nn.Linear(d_model, d_model)  # [D, D]
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)  # [D, D]
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: Tensor,      # [B, L, D] or None (for self-attn, all three are same)
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        B, L, D = query.shape
        num_heads = self.num_heads
        d_k = self.d_k
        
        # ---- Step 1: 线性投影得到 Q, K, V ----
        # 每个都是 [B, L, D]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # ---- Step 2: 分成多个头 ----
        # 将 D 维度分成 (num_heads, d_k)
        # [B, L, D] -> [B, L, num_heads, d_k] -> [B, num_heads, L, d_k]
        Q = Q.view(B, L, num_heads, d_k).transpose(1, 2)   # [B, h, L, d_k]
        K = K.view(B, K.size(1), num_heads, d_k).transpose(1, 2)  # [B, h, L_k, d_k]
        V = V.view(B, V.size(1), num_heads, d_k).transpose(1, 2)  # [B, h, L_v, d_k]
        
        print(f"\n  Q/K/V Shape: [B={B}, h={num_heads}, L={L}, d_k={d_k}]")
        
        # ---- Step 3: 计算注意力分数 ----
        # QK^T: [B,h,L,d_k] @ [B,h,d_k,L] -> [B,h,L,L]
        # 除以 sqrt(d_k) 防止点积过大导致梯度消失
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        print(f"  注意力分数 Shape: {scores.shape}  # [B, h, L, L]")
        
        # ---- Step 4: 应用 Mask (如果需要) ----
        if mask is not None:
            # 将 mask 广播到与 scores 相同形状
            # causal mask: 上三角位置填 -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
            print(f"  应用 Mask 后的分数 (部分): {scores[0, 0, 0, :5]}...  # 上三角-inf")
        
        # ---- Step 5: Softmax 得到注意力权重 ----
        attn_weights = F.softmax(scores, dim=-1)  # [B, h, L, L]
        attn_weights = self.dropout(attn_weights)
        
        # ---- Step 6: 加权求和得到注意力输出 ----
        # attn_weights @ V: [B,h,L,L] @ [B,h,L,d_k] -> [B,h,L,d_k]
        context = torch.matmul(attn_weights, V)
        print(f"  注意力输出 Shape: {context.shape}  # [B, h, L, d_k]")
        
        # ---- Step 7: 合并多个头 ----
        # [B, h, L, d_k] -> [B, L, h, d_k] -> [B, L, D]
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, L, D)
        
        # ---- Step 8: 最终线性投影 ----
        output = self.W_o(context)
        print(f"  最终输出 Shape: {output.shape}  # [B, L, D]")
        
        return output  # [B, L, D]


def causal_mask(seq_len: int, device: torch.device = torch.device('cpu')) -> Tensor:
    """
    创建因果掩码 (causal mask)
    确保位置 t 只能看到 1 到 t-1 的信息
    
    返回: [1, 1, seq_len, seq_len] 的上三角为 0 (被遮蔽)
    """
    # 上三角为 0 (会被设为 -inf)
    # 下三角为 1 (保持可见)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def demo_self_attention():
    """演示自注意力机制的维度变化"""
    print("\n" + "="*60)
    print("2. MULTI-HEAD SELF-ATTENTION - 演示")
    print("="*60)
    
    B, L = 2, 4  # batch=2, seq_len=4
    D = cfg.d_model  # 512
    
    # 模拟输入
    x = torch.randn(B, L, D)
    print(f"输入 Shape: {x.shape}  # [B, L, D]")
    print(f"  D={D}, h={cfg.num_heads}, d_k=D/h={D//cfg.num_heads}")
    
    # 创建自注意层 (Q=K=V=x)
    attn = MultiHeadAttention(D, cfg.num_heads)
    
    # 无 mask 的自注意力
    print("\n--- Self-Attention (无 Mask) ---")
    output = attn(query=x, key=x, value=x, mask=None)
    print(f"输出 Shape: {output.shape}  # [B, L, D], 维度保持不变")
    
    # 有 causal mask 的自注意力
    print("\n--- Masked Self-Attention (Causal Mask) ---")
    mask = causal_mask(L)
    print(f"Mask Shape: {mask.shape}  # [1, 1, L, L]")
    print(f"Causal Mask (1=可见, 0=遮蔽):\n{mask[0, 0]}")
    
    output_masked = attn(query=x, key=x, value=x, mask=mask)
    print(f"输出 Shape: {output_masked.shape}")
    
    return output_masked


# ============================================================
# 3. 交叉注意力 (Cross Attention)
# ============================================================

def demo_cross_attention():
    """演示 Encoder-Decoder 交叉注意力"""
    print("\n" + "="*60)
    print("3. CROSS ATTENTION - 演示")
    print("="*60)
    
    B = 2
    L_src = 6  # 源序列长度 (Encoder 输出)
    L_tgt = 4  # 目标序列长度 (Decoder 输入)
    D = cfg.d_model  # 512
    
    # Encoder 输出 (作为 K, V)
    encoder_output = torch.randn(B, L_src, D)
    print(f"Encoder 输出 (K, V): {encoder_output.shape}  # [B, L_src, D]")
    
    # Decoder 隐藏状态 (作为 Q)
    decoder_hidden = torch.randn(B, L_tgt, D)
    print(f"Decoder 隐藏状态 (Q): {decoder_hidden.shape}  # [B, L_tgt, D]")
    
    # 交叉注意力
    cross_attn = MultiHeadAttention(D, cfg.num_heads)
    
    # Q 来自 Decoder, K/V 来自 Encoder
    output = cross_attn(query=decoder_hidden, key=encoder_output, value=encoder_output, mask=None)
    
    print(f"\nCross Attention 输出: {output.shape}")
    print(f"  Q 序列长度 L_tgt={L_tgt}")
    print(f"  K/V 序列长度 L_src={L_src}")
    print(f"  输出序列长度 = Q 的长度 = {L_tgt}")
    
    # 注意力矩阵形状: [B, h, L_tgt, L_src]
    print(f"  注意力矩阵 Shape: [B, {cfg.num_heads}, {L_tgt}, {L_src}]")
    print(f"  每个目标位置可以关注源序列所有位置")
    
    return output


# ============================================================
# 4. 前馈神经网络 (Feed Forward Network)
# ============================================================

class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise FFN)
    
    输入: [B, L, D]
    输出: [B, L, D]
    
    结构: Linear → ReLU → Linear
    维度: D → d_ff (2048) → D
    
    公式: FFN(x) = max(0, xW₁+b₁)W₂+b₂
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # [D, d_ff]
        self.fc2 = nn.Linear(d_ff, d_model)     # [d_ff, D]
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        print(f"\n  FFN 输入 Shape: {x.shape}")
        
        # 第一层线性变换 + ReLU
        x = self.fc1(x)  # [B, L, d_ff]
        print(f"  W1 投影后 Shape: {x.shape}  # {cfg.d_model} → {cfg.d_ff}")
        
        x = F.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 第二层线性变换
        x = self.fc2(x)  # [B, L, D]
        print(f"  W2 投影后 Shape: {x.shape}  # {cfg.d_ff} → {cfg.d_model}")
        
        return x  # [B, L, D]


def demo_ffn():
    """演示 FFN 的维度变化"""
    print("\n" + "="*60)
    print("4. FEED FORWARD NETWORK - 演示")
    print("="*60)
    
    B, L = 2, 5
    D = cfg.d_model  # 512
    d_ff = cfg.d_ff  # 2048
    
    x = torch.randn(B, L, D)
    print(f"输入 Shape: {x.shape}  # [B, L, D]")
    
    ffn = FeedForward(D, d_ff)
    output = ffn(x)
    
    print(f"\nFFN 输出 Shape: {output.shape}  # [B, L, D]")
    print(f"维度变化: {D} → {d_ff} → {D}")
    
    return output


# ============================================================
# 5. 层归一化 (Layer Normalization)
# ============================================================

class LayerNorm(nn.Module):
    """
    层归一化
    
    输入: [B, L, D]
    输出: [B, L, D]
    
    公式: LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
    
    其中:
        μ = mean(x, dim=-1)     # 在 D 维度上求均值
        σ² = var(x, dim=-1)    # 在 D 维度上求方差
    
    与 BatchNorm 的区别:
        - BatchNorm: 在 batch 维度归一化
        - LayerNorm: 在特征维度归一化
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        mean = x.mean(dim=-1, keepdim=True)   # [B, L, 1]
        std = x.std(dim=-1, keepdim=True)     # [B, L, 1]
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和偏移
        return self.gamma * x_norm + self.beta


class AddAndNorm(nn.Module):
    """
    残差连接 + 层归一化
    
    输出 = LayerNorm(x + Sublayer(x))
    
    残差连接好处:
        1. 梯度可以直接回传，避免深层网络梯度消失
        2. 让网络更容易学习恒等映射
    """
    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        # 残差连接
        x = x + self.dropout(sublayer_output)
        # 层归一化
        return self.layernorm(x)


def demo_add_and_norm():
    """演示 Add & LayerNorm"""
    print("\n" + "="*60)
    print("5. ADD & LAYER NORMALIZATION - 演示")
    print("="*60)
    
    B, L, D = 2, 5, 512
    
    # 模拟输入和子层输出 (例如 attention 的输出)
    x = torch.randn(B, L, D)
    sublayer_out = torch.randn(B, L, D)
    
    print(f"输入 x Shape: {x.shape}")
    print(f"子层输出 Shape: {sublayer_out.shape}")
    print(f"维度相同才能做残差连接")
    
    # 手动演示
    x_with_residual = x + sublayer_out
    print(f"\n残差连接后 Shape: {x_with_residual.shape}")
    
    # LayerNorm
    layernorm = LayerNorm(D)
    output = layernorm(x_with_residual)
    print(f"LayerNorm 后 Shape: {output.shape}")
    
    return output


# ============================================================
# 6. 完整的 Encoder Layer
# ============================================================

class EncoderLayer(nn.Module):
    """
    单个 Encoder 层
    
    输入: [B, L, D]
    输出: [B, L, D]
    
    结构:
        1. Multi-Head Self-Attention
        2. Add & LayerNorm
        3. Feed Forward Network
        4. Add & LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # ---- Self-Attention + Add&Norm ----
        print(f"\n  Encoder Layer 输入: {x.shape}")
        
        # Self-Attention: Q=K=V=x
        attn_output = self.self_attn(x, x, x, mask)
        print(f"  Self-Attn 输出: {attn_output.shape}")
        
        # Add & Norm (残差连接)
        x = self.norm1(x + self.dropout1(attn_output))
        print(f"  Add&Norm 后: {x.shape}")
        
        # ---- FFN + Add&Norm ----
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        print(f"  FFN + Add&Norm 后: {x.shape}")
        
        return x  # [B, L, D]


def demo_encoder_layer():
    """演示 Encoder Layer"""
    print("\n" + "="*60)
    print("6. ENCODER LAYER - 演示")
    print("="*60)
    
    B, L, D = 2, 4, 512
    
    x = torch.randn(B, L, D)
    print(f"输入 Shape: {x.shape}")
    
    encoder_layer = EncoderLayer(D, cfg.num_heads, cfg.d_ff)
    
    # 创建 mask (如果需要)
    mask = None  # 或者 causal_mask(L)
    
    output = encoder_layer(x, mask)
    print(f"\nEncoder Layer 输出: {output.shape}")
    print(f"输入输出维度相同: {L} → {L}, {D} → {D}")
    
    return output


# ============================================================
# 7. 完整的 Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):
    """
    单个 Decoder 层
    
    输入: [B, L_tgt, D]
    输出: [B, L_tgt, D]
    
    结构:
        1. Masked Self-Attention (因果 attention)
        2. Add & LayerNorm
        3. Cross Attention (Q 来自 decoder, K/V 来自 encoder)
        4. Add & LayerNorm
        5. Feed Forward Network
        6. Add & LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.masked_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        
        # Cross Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: Tensor,                    # Decoder 输入
        encoder_output: Tensor,       # Encoder 输出 (K, V)
        tgt_mask: Optional[Tensor] = None,    # Target 因果 mask
        src_mask: Optional[Tensor] = None     # Source padding mask (可选)
    ) -> Tensor:
        print(f"\n  Decoder Layer 输入: {x.shape}")
        
        # ---- Masked Self-Attention + Add&Norm ----
        # Q = K = V = x, 需要 causal mask
        masked_attn_out = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(masked_attn_out))
        print(f"  Masked Self-Attn + Add&Norm: {x.shape}")
        
        # ---- Cross Attention + Add&Norm ----
        # Q 来自 decoder, K/V 来自 encoder output
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        print(f"  Cross Attn + Add&Norm: {x.shape}")
        
        # ---- FFN + Add&Norm ----
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        print(f"  FFN + Add&Norm: {x.shape}")
        
        return x


def demo_decoder_layer():
    """演示 Decoder Layer"""
    print("\n" + "="*60)
    print("7. DECODER LAYER - 演示")
    print("="*60)
    
    B = 2
    L_tgt = 4  # 目标序列长度
    L_src = 6  # 源序列长度
    D = 512
    
    # Decoder 输入
    x = torch.randn(B, L_tgt, D)
    print(f"Decoder 输入 Shape: {x.shape}")
    
    # Encoder 输出 (来自 encoder)
    encoder_output = torch.randn(B, L_src, D)
    print(f"Encoder 输出 Shape: {encoder_output.shape}")
    
    decoder_layer = DecoderLayer(D, cfg.num_heads, cfg.d_ff)
    
    # 创建因果 mask
    tgt_mask = causal_mask(L_tgt)
    
    output = decoder_layer(x, encoder_output, tgt_mask)
    print(f"\nDecoder Layer 输出: {output.shape}")
    
    return output


# ============================================================
# 8. 完整的 Transformer 模型
# ============================================================

class TransformerEncoder(nn.Module):
    """完整的 Transformer Encoder"""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Embedding + 位置编码
        x = self.embedding(x)
        x = self.pos_enc(x)
        
        # 通过多层 Encoder
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """完整的 Transformer Decoder"""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
    
    def forward(
        self, 
        x: Tensor, 
        encoder_output: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Embedding + 位置编码
        x = self.embedding(x)
        x = self.pos_enc(x)
        
        # 通过多层 Decoder
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x


class Transformer(nn.Module):
    """
    完整的 Transformer 模型 (Encoder + Decoder)
    
    用于序列到序列的任务，如机器翻译
    """
    def __init__(self, config: Config):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            config.vocab_size, config.d_model, config.num_heads,
            config.num_layers, config.d_ff, config.max_len, config.dropout
        )
        
        self.decoder = TransformerDecoder(
            config.vocab_size, config.d_model, config.num_heads,
            config.num_layers, config.d_ff, config.max_len, config.dropout
        )
        
        # 输出层: 将 Decoder 输出映射到词表维度
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(
        self, 
        src: Tensor,     # 源序列 [B, L_src]
        tgt: Tensor,     # 目标序列 [B, L_tgt] (右移一位)
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        # ---- Encoder ----
        encoder_output = self.encoder(src, src_mask)
        print(f"\nEncoder 输出: {encoder_output.shape}  # [B, L_src, D]")
        
        # ---- Decoder ----
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        print(f"Decoder 输出: {decoder_output.shape}  # [B, L_tgt, D]")
        
        # ---- Output Projection ----
        # 将隐藏状态映射到词表维度，得到每个位置的 logit
        logits = self.output_proj(decoder_output)
        print(f"Output logits: {logits.shape}  # [B, L_tgt, vocab_size]")
        
        return logits  # [B, L_tgt, vocab_size]


def demo_transformer():
    """演示完整的 Transformer 模型"""
    print("\n" + "="*60)
    print("8. COMPLETE TRANSFORMER - 演示")
    print("="*60)
    
    B = 2
    L_src = 6  # 源序列长度
    L_tgt = 4  # 目标序列长度
    
    # 源序列 (例如中文)
    src = torch.randint(0, cfg.vocab_size, (B, L_src))
    print(f"源序列 (输入): {src.shape}  # [B, L_src]")
    
    # 目标序列 (例如英文, 右移一位)
    tgt = torch.randint(0, cfg.vocab_size, (B, L_tgt))
    print(f"目标序列 (输出): {tgt.shape}  # [B, L_tgt]")
    
    # Transformer 模型
    model = Transformer(cfg)
    
    # Masks
    tgt_mask = causal_mask(L_tgt)
    
    # Forward
    print("\n--- Forward Pass ---")
    logits = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"\n最终输出 Shape: {logits.shape}")
    print(f"每个位置输出 {cfg.vocab_size} 维的 logit (未归一化的分数)")
    
    # 得到概率分布
    probs = F.softmax(logits, dim=-1)  # [B, L_tgt, vocab_size]
    print(f"概率分布 Shape: {probs.shape}")
    
    # 取每个位置概率最高的 token
    predictions = torch.argmax(probs, dim=-1)  # [B, L_tgt]
    print(f"预测的 Token IDs: {predictions.shape}")
    
    return logits


# ============================================================
# 主函数 - 运行所有演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFORMER ARCHITECTURE - PyTorch 实现详解")
    print("=" * 60)
    print(f"配置: d_model={cfg.d_model}, h={cfg.num_heads}, "
          f"d_ff={cfg.d_ff}, layers={cfg.num_layers}")
    
    # 逐个演示各个模块
    demo_input_encoding()
    demo_self_attention()
    demo_cross_attention()
    demo_ffn()
    demo_add_and_norm()
    demo_encoder_layer()
    demo_decoder_layer()
    demo_transformer()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)