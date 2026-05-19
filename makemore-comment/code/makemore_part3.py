"""
Makemore Part 3 — 激活函数与 BatchNorm

本代码演示激活函数统计和 BatchNorm 原理。

来源: https://github.com/karpathy/makemore
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# 激活函数统计
# ============================================================================

def print_activations_stats(name, x):
    """
    打印激活值的统计信息，用于诊断训练问题

    Args:
        name: 层名称
        x: 激活值张量
    """
    # 计算统计量
    mean = x.mean().item()
    std = x.std().item()
    min_val = x.min().item()
    max_val = x.max().item()

    print(f"{name}:")
    print(f"  mean={mean:.4f}, std={std:.4f}")
    print(f"  min={min_val:.4f}, max={max_val:.4f}")


# ============================================================================
# BatchNorm 实现
# ============================================================================

class BatchNorm(nn.Module):
    """
    Batch Normalization

    公式: y = gamma * (x - mean) / sqrt(var + eps) + beta

    作用:
    1. 标准化输入，稳定训练
    2. 允许更高的学习率
    3. 提供正则化效果

    维度: (B, T, C) → (B, T, C)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 移动平均统计量（用于推理）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, training=True):
        """
        前向传播

        Args:
            x: (B, T, C) 输入
            training: 是否训练模式

        Returns:
            y: (B, T, C) 标准化后的输出
        """
        if training:
            # 计算 batch 统计量
            # x.mean(dim=(0,1)) 计算每个特征的均值
            mean = x.mean(dim=(0, 1))  # (C,)
            var = x.var(dim=(0, 1), unbiased=False)  # (C,)

            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 使用移动平均（推理模式）
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_norm = (x - mean.unsqueeze(0).unsqueeze(1)) / torch.sqrt(var.unsqueeze(0).unsqueeze(1) + self.eps)

        # 缩放和平移
        y = self.gamma.unsqueeze(0).unsqueeze(1) * x_norm + self.beta.unsqueeze(0).unsqueeze(1)

        return y


# ============================================================================
# 梯度诊断
# ============================================================================

def diagnose_gradients(model):
    """
    诊断模型梯度，检测梯度消失/爆炸问题

    Args:
        model: 神经网络模型

    Returns:
        dict: 每层参数的梯度范数
    """
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

            # 打印异常梯度
            if grad_norm < 1e-6:
                print(f"[WARNING] {name}: gradient near zero ({grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"[WARNING] {name}: gradient exploding ({grad_norm:.2e})")

    return grad_norms


# ============================================================================
# 权重初始化
# ============================================================================

def init_weights(model, init_type='xavier'):
    """
    初始化模型权重

    Args:
        model: 神经网络模型
        init_type: 'xavier', 'kaiming', 'normal'
    """
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(param)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            nn.init.zeros_(param)


# ============================================================================
# 示例: 使用 BatchNorm 的 MLP
# ============================================================================

class MLPWithBatchNorm(nn.Module):
    """
    带 BatchNorm 的 MLP 示例

    维度变化:
    ┌─────────────────────────────────────────────────────────────┐
    │  输入 x: (B, T, n_embd)                                    │
    │      ↓                                                     │
    │  Linear: n_embd → n_hidden                                │
    │      ↓                                                     │
    │  BatchNorm: 标准化                                        │
    │      ↓                                                     │
    │  ReLU: 激活函数                                            │
    │      ↓                                                     │
    │  Linear: n_hidden → vocab_size                            │
    │      ↓                                                     │
    │  输出 logits: (B, T, vocab_size)                          │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_embd, n_hidden, vocab_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            BatchNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, vocab_size)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================================
# 训练诊断示例
# ============================================================================

def training_diagnosis_example():
    """
    训练诊断示例代码
    """
    # 创建模型
    model = MLPWithBatchNorm(n_embd=64, n_hidden=128, vocab_size=100)

    # 初始化权重
    init_weights(model, init_type='normal')

    # 模拟前向传播
    x = torch.randn(32, 10, 64)  # (B, T, n_embd)

    # 打印输入统计
    print_activations_stats("input", x)

    # 前向传播
    logits = model(x)

    # 打印输出统计
    print_activations_stats("output", logits)

    # 模拟损失和反向传播
    targets = torch.randint(0, 100, (32, 10))
    loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))

    # 反向传播
    loss.backward()

    # 诊断梯度
    grad_norms = diagnose_gradients(model)

    print("\n=== 梯度范数 ===")
    for name, norm in sorted(grad_norms.items()):
        print(f"{name}: {norm:.6f}")


if __name__ == '__main__':
    training_diagnosis_example()