"""
Makemore Part 5 — WaveNet 与 CNN

本代码演示卷积神经网络、膨胀卷积和层次化结构。

来源: https://github.com/karpathy/makemore
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# 第一部分: 基础卷积
# ============================================================================

class Conv1d(nn.Module):
    """
    一维卷积层

    公式: y[i] = sum(w[k] * x[i + stride*k] for k in range(kernel_size))

    维度变化:
    ┌─────────────────────────────────────────────────────────────┐
    │  输入: (B, C_in, T)                                        │
    │      ↓                                                     │
    │  卷积: C_in * kernel_size → C_out                        │
    │      ↓                                                     │
    │  输出: (B, C_out, T_out)                                   │
    │                                                      │
    │  T_out = floor((T + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 权重和偏置
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, C_in, T)

        Returns:
            y: (B, C_out, T_out)
        """
        return F.conv1d(x, self.weight, self.bias,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation)


# ============================================================================
# 第二部分: 膨胀卷积 (Dilated Convolution)
# ============================================================================

class DilatedConv1d(nn.Module):
    """
    膨胀卷积层

    原理: 通过在输入元素之间插入空格来扩大感受野

    感受野计算:
    - 普通卷积 (k=3, d=1): 感受野 = k = 3
    - 膨胀卷积 (k=3, d=2): 感受野 = k + (k-1)*(d-1) = 3 + 2*1 = 5
    - 膨胀卷积 (k=3, d=4): 感受野 = 3 + 2*3 = 9

    维度变化: 输入 T不变，输出 T 不变（same padding）
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.dilation = dilation

        # 使用膨胀卷积
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,  # 保持长度
            dilation=dilation
        )

    def forward(self, x):
        return self.conv(x)


class WaveNetBlock(nn.Module):
    """
    WaveNet 残差块

    结构:
    ┌─────────────────────────────────────────────────────────────┐
    │  输入 x                                                    │
    │      ↓                                                     │
    │  1x1 卷积 (gate)                                          │
    │      ↓                                                     │
    │  分支:                                                     │
    │    ├─ tanh (门控激活)                                      │
    │    └─ sigmoid (门控激活)                                    │
    │      ↓                                                     │
    │  Multiply (门控)                                           │
    │      ↓                                                     │
    │  1x1 卷积                                                  │
    │      ↓                                                     │
    │  残差连接: x + output                                      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, residual_channels, gate_channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation

        # 门控卷积
        self.conv = nn.Conv1d(
            residual_channels, gate_channels * 2,  # gate channels for both tanh and sigmoid
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )

        # 1x1 卷积用于残差连接
        self.residual_conv = nn.Conv1d(gate_channels, residual_channels, 1)

        # 1x1 卷积用于跳跃连接
        self.skip_conv = nn.Conv1d(gate_channels, residual_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T)

        Returns:
            output: 残差后的输出
            skip: 跳跃连接输出
        """
        # 门控卷积
        conv_out = self.conv(x)

        # 分离 tanh 和 sigmoid 门
        # 假设 gate_channels = residual_channels
        half = conv_out.size(1) // 2
        tanh_out = torch.tanh(conv_out[:, :half, :])
        sigmoid_out = torch.sigmoid(conv_out[:, half:, :])

        # 门控激活
        gated = tanh_out * sigmoid_out

        # 残差连接
        residual = self.residual_conv(gated)
        x = x + residual  # 残差: x = x + F(x)

        # 跳跃连接
        skip = self.skip_conv(gated)

        return x, skip


# ============================================================================
# 第三部分: 因果卷积 (Causal Convolution)
# ============================================================================

class CausalConv1d(nn.Module):
    """
    因果卷积层

    确保 t 时刻的输出只依赖 t 时刻及之前的输入

    原理: 填充到左边，使得输出向右偏移
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # 计算填充: 保持因果性
        # 输出 y[t] 依赖输入 x[0..t]
        # 如果 kernel_size = k，则需要填充 k-1 个位置到左边
        padding = (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T)

        Returns:
            y: (B, C, T) - 因果卷积输出
        """
        conv_out = self.conv(x)

        # 移除左边填充，恢复因果性
        # 填充了 kernel_size - 1 个位置，需要移除
        return conv_out[:, :, :-self.kernel_size + 1] if self.kernel_size > 1 else conv_out


# ============================================================================
# 第四部分: WaveNet 模型
# ============================================================================

class WaveNet(nn.Module):
    """
    WaveNet 模型

    结构:
    ┌─────────────────────────────────────────────────────────────┐
    │  输入嵌入                                                  │
    │      ↓                                                     │
    │  初始因果卷积                                              │
    │      ↓                                                     │
    │  膨胀卷积堆叠 (dilation: 1, 2, 4, 8, ...)                   │
    │      ↓                                                     │
    │  1x1 卷积                                                  │
    │      ↓                                                     │
    │  ReLU                                                      │
    │      ↓                                                     │
    │  1x1 卷积 → 输出 logits                                   │
    └─────────────────────────────────────────────────────────────┘

    感受野: 1 + k + k*d1 + k*d2 + ... = 指数增长
    """

    def __init__(self, vocab_size, n_embd, n_channels, n_layers, kernel_size=3):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_channels = n_channels
        self.n_layers = n_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, n_embd)

        # 初始因果卷积
        self.input_conv = CausalConv1d(n_embd, n_channels, kernel_size)

        # 膨胀卷积块
        self.blocks = nn.ModuleList()
        dilations = [2 ** i for i in range(n_layers)]

        for dilation in dilations:
            self.blocks.append(
                WaveNetBlock(n_channels, n_channels, kernel_size, dilation)
            )

        # 输出层
        self.output_conv1 = nn.Conv1d(n_channels, n_channels, 1)
        self.output_conv2 = nn.Conv1d(n_channels, vocab_size, 1)

    def forward(self, x, target=None):
        """
        Args:
            x: (B, T) 字符索引
            target: (B, T) 目标索引（用于计算损失）

        Returns:
            logits: (B, T, vocab_size)
            loss: 交叉熵损失（如果提供了 target）
        """
        B, T = x.size()

        # 嵌入
        x = self.embedding(x)  # (B, T, n_embd)

        # 转置为 (B, C, T)
        x = x.transpose(1, 2)

        # 初始卷积
        x = self.input_conv(x)

        # 膨胀卷积块
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # 跳过连接求和
        skip_sum = sum(skip_connections)

        # 输出卷积
        x = F.relu(skip_sum)
        x = self.output_conv1(x)
        x = F.relu(x)
        logits = self.output_conv2(x)  # (B, vocab_size, T)

        # 转置回来: (B, T, vocab_size)
        logits = logits.transpose(1, 2)

        # 计算损失
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.view(-1),
                ignore_index=-1
            )

        return logits, loss


# ============================================================================
# 第五部分: CNN vs 其他模型对比
# ============================================================================

def compare_models():
    """
    CNN vs RNN vs Transformer 对比

    | 特性       | CNN          | RNN          | Transformer   |
    |------------|--------------|--------------|---------------|
    | 并行性     | 高           | 低           | 高            |
    | 感受野     | 有限         | 全局         | 全局          |
    | 长依赖     | 难           | 容易(梯度)   | 容易          |
    | 计算量     | O(n)         | O(n)         | O(n²)         |
    | 内存       | O(n)         | O(n)         | O(n²)         |
    """

    print("=== CNN vs RNN vs Transformer 对比 ===")
    print()
    print("| 特性       | CNN          | RNN          | Transformer   |")
    print("|------------|--------------|--------------|---------------|")
    print("| 并行性     | 高           | 低           | 高            |")
    print("| 感受野     | 有限         | 全局         | 全局          |")
    print("| 长依赖     | 难           | 容易(梯度)   | 容易          |")
    print("| 计算量     | O(n)         | O(n)         | O(n²)         |")
    print("| 内存       | O(n)         | O(n)         | O(n²)         |")


def dilation_example():
    """膨胀卷积感受野示例"""
    print()
    print("=== 膨胀卷积感受野示例 ===")
    print()

    kernel_size = 3

    print(f"卷积核大小: {kernel_size}")
    print()

    for dilation in [1, 2, 4, 8]:
        # 感受野 = k + (k-1)*(d-1) = 3 + 2*(d-1)
        receptive_field = kernel_size + (kernel_size - 1) * (dilation - 1)
        print(f"  膨胀率 d={dilation}: 感受野 = {receptive_field}")

    print()
    print("堆叠多个膨胀卷积后的总感受野:")
    print("  d=[1,2,4,8]: 3 + 6 + 12 + 24 = 45 (理论值)")
    print("  实际: 1 + 3 + 6 + 12 + 24 = 46 (含初始层)")


if __name__ == '__main__':
    compare_models()
    dilation_example()

    # 测试 WaveNet
    print()
    print("=== WaveNet 测试 ===")

    model = WaveNet(
        vocab_size=100,
        n_embd=32,
        n_channels=64,
        n_layers=4,
        kernel_size=3
    )

    # 随机输入
    x = torch.randint(0, 100, (8, 20))  # (B, T)
    target = torch.randint(0, 100, (8, 20))

    # 前向传播
    logits, loss = model(x, target)

    print(f"输入: (B={x.size(0)}, T={x.size(1)})")
    print(f"输出 logits: {logits.shape}")
    print(f"损失: {loss.item():.4f}")