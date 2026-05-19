"""
micrograd — 自动微分引擎

本包是 Andrej Karpathy 的 micrograd 课程实现，包含：
- engine: 自动求导核心（Value 类）
- nn: 神经网络层定义

使用示例：
    from micrograd.engine import Value

    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x ** 2
    z.backward()

    print(f"dz/dx = {x.grad}")  # 7

或者：
    from micrograd.nn import MLP

    model = MLP(2, [8, 4, 1])
    output = model([1.0, 2.0])
"""

from .engine import Value
from .nn import Module, Neuron, Layer, MLP

__all__ = ["Value", "Module", "Neuron", "Layer", "MLP"]
