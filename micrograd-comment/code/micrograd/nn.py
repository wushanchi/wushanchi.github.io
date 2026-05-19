"""
micrograd 神经网络模块 — 基础网络层定义

本模块实现了神经网络的基本组件：
- Module: 所有网络层的基类
- Neuron: 单神经元
- Layer: 全连接层
- MLP: 多层感知机

来源: https://github.com/karpathy/micrograd

============ 层级结构 ============

Module (基类)
    │
    ├── Neuron(nin)      # 单神经元: y = ReLU(w·x + b)
    │       └── 参数: nin 个权重 + 1 个偏置
    │
    ├── Layer(nin, nout) # 全连接层: nout 个神经元并行
    │       └── 参数: nin×nout 个权重 + nout 个偏置
    │
    └── MLP(nin, nouts)  # 多层感知机: 多个 Layer 堆叠
            └── 参数: 所有 Layer 参数之和

============ 维度变换 ============

Neuron:
    输入: x = [x₁, x₂, ..., xₙᵢₙ]  (nin维向量)
    权重: w = [w₁, w₂, ..., wₙᵢₙ]  (nin维向量)
    偏置: b (标量)
    计算: act = Σ(wᵢ × xᵢ) + b = w · x + b
    输出: out = ReLU(act) 或 act

Layer:
    输入: x = [x₁, x₂, ..., xₙᵢₙ]  (nin维向量)
    输出: y = [y₁, y₂, ..., yₙᵒᵤₜ] (nout维向量)
    每个 yᵢ = Neuronᵢ(x)

MLP:
    由多个 Layer 堆叠而成
    除最后一层外都使用 ReLU 激活
"""

import random
from micrograd.engine import Value


class Module:
    """
    神经网络模块基类

    所有网络层（Neuron、Layer、MLP）都继承自 Module。
    Module 定义了两个核心接口：
    - parameters(): 返回所有可学习参数
    - zero_grad(): 清除所有参数的梯度

    设计模式参考 PyTorch 的 torch.nn.Module。
    """

    def zero_grad(self):
        """
        清除所有参数的梯度

        在每次训练迭代开始时调用，将所有参数的 grad 设为0。
        注意：使用 = 赋值而不是 +=，因为 grad 在反向传播时是累加的。

        Example:
            model.zero_grad()  # 训练循环开始时
            loss.backward()     # 反向传播
            optimizer.step()    # 参数更新
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        返回所有可学习参数

        子类需要重写此方法，返回包含所有可学习参数的列表。

        Returns:
            list: Value 参数列表
        """
        return []


class Neuron(Module):
    """
    单神经元 — 神经网络的基本单元

    实现: y = ReLU(w · x + b) 或 y = w · x + b

    神经元是神经网络的基本构建块，模拟生物神经元的工作方式：
    - 接收多个输入信号
    - 每个输入有对应的权重（表示该输入的重要程度）
    - 加上偏置后通过激活函数输出

    参数维度：
        输入: x (nin维向量)
        权重: w (nin维向量)
        偏置: b (标量)
        输出: y (标量或向量，取决于 nonlin)
    """

    def __init__(self, nin, nonlin=True):
        """
        初始化神经元

        Args:
            nin: 输入特征的维度（输入元素个数）
            nonlin: 是否使用激活函数，默认 True（使用 ReLU）
                   最后一层通常设为 False（线性输出）

        Example:
            # 创建一个 3 输入的神经元（带 ReLU）
            neuron = Neuron(3)      # y = ReLU(w·x + b)

            # 创建一个线性神经元（无激活）
            neuron = Neuron(3, nonlin=False)  # y = w·x + b
        """
        # 初始化权重向量：nin 个权重，每个从 [-1, 1] 均匀分布随机采样
        # 使用随机权重是为了打破对称性，让不同神经元学习不同特征
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # 初始化偏置：默认为 0
        self.b = Value(0)

        # 是否使用非线性激活（ReLU）
        # 隐藏层通常为 True，输出层根据任务决定
        self.nonlin = nonlin

    def __call__(self, x):
        """
        前向传播计算

        计算并返回神经元的输出。

        计算过程：
            act = Σ(wᵢ × xᵢ) + b = w · x + b
            out = ReLU(act) if self.nonlin else act

        Args:
            x: 输入向量（列表或数组），长度为 nin

        Returns:
            Value: 神经元输出

        Example:
            neuron = Neuron(3)
            output = neuron([1.0, 2.0, 3.0])  # 返回 Value 对象
        """
        # 计算加权和：act = w·x + b
        # sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        # 遍历权重和输入的配对，相乘后求和，最后加上偏置
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # 如果启用激活函数，应用 ReLU
        # ReLU(x) = max(0, x)，将负数置为0
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        返回神经元的所有可学习参数

        Returns:
            list: [w₁, w₂, ..., wₙᵢₙ, b] — 权重和偏置的列表
        """
        return self.w + [self.b]  # 权重列表 + 偏置

    def __repr__(self):
        """
        字符串表示，用于调试和打印

        Returns:
            str: 神经元类型和输入维度
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    全连接层 — 包含多个神经元的层

    全连接层（Fully Connected Layer / Dense Layer）由多个神经元组成，
    每个神经元独立处理相同的输入，产生一个输出。

    维度变换：
        输入: x (nin维向量)
        输出: y (nout维向量)
        其中 y[i] = Neuron_i(x)

    参数数量：
        nin × nout（权重）+ nout（偏置）= nout × (nin + 1)
    """

    def __init__(self, nin, nout, **kwargs):
        """
        初始化全连接层

        Args:
            nin: 输入特征维度
            nout: 输出神经元个数
            **kwargs: 传递给 Neuron 的额外参数（如 nonlin=False）

        Example:
            # 创建一个 3 输入、4 输出的层
            layer = Layer(3, 4)

            # 创建一个最后一层（无激活）
            layer = Layer(8, 1, nonlin=False)
        """
        # 创建 nout 个神经元，每个神经元有 nin 个输入
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        前向传播计算

        并行计算所有神经元的输出。

        Args:
            x: 输入向量（nin维）

        Returns:
            Value 或 list:
                - 如果只有 1 个神经元，返回 Value
                - 如果有多个神经元，返回 Value 列表
        """
        # 并行计算每个神经元的输出
        out = [n(x) for n in self.neurons]

        # 如果只有一个神经元，返回单个 Value 而不是列表
        # 这样做是为了保持 API 的一致性
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        返回该层所有参数

        Returns:
            list: 所有 nout 个神经元的参数（展平为一维列表）
        """
        # 使用列表推导式展开所有神经元的参数
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        字符串表示

        Returns:
            str: 层中所有神经元的描述
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    多层感知机（Multi-Layer Perceptron）

    MLP 由多个全连接层堆叠而成，除最后一层外每层都使用 ReLU 激活。

    网络结构示例：
        MLP(2, [8, 4, 1])
            输入层 (2)
                ↓
            Layer 1 (2→8) + ReLU
                ↓
            Layer 2 (8→4) + ReLU
                ↓
            Layer 3 (4→1) + Linear (输出层)
                ↓
            输出 (1)

    参数数量计算：
        Layer 1: 2×8 + 8 = 24
        Layer 2: 8×4 + 4 = 36
        Layer 3: 4×1 + 1 = 5
        总计: 65 个参数
    """

    def __init__(self, nin, nouts):
        """
        初始化 MLP

        Args:
            nin: 输入层维度（输入特征的个数）
            nouts: 各层的输出维度列表，如 [8, 4, 1] 表示：
                   - 隐藏层 1: 8 个神经元 + ReLU
                   - 隐藏层 2: 4 个神经元 + ReLU
                   - 输出层: 1 个神经元（无激活）

        Example:
            # 创建一个 2 输入、隐藏层 [16, 8]、1 输出的 MLP
            model = MLP(2, [16, 8, 1])

            # 打印网络结构
            print(model)
            # MLP of [Layer of [ReLUNeuron(2), ...×16],
            #         Layer of [ReLUNeuron(16), ...×8],
            #         Layer of [LinearNeuron(8)]]
        """
        # 构建完整的维度列表：[nin, nouts[0], nouts[1], ..., nouts[-1]]
        # 例如：nin=2, nouts=[8,4,1] -> sz=[2,8,4,1]
        sz = [nin] + nouts

        # 创建多层网络
        # 除最后一层外都使用 ReLU 激活
        # nonlin = i != len(nouts) - 1 意味着：
        # - 隐藏层 (i < len(nouts)-1): nonlin=True
        # - 输出层 (i == len(nouts)-1): nonlin=False
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """
        前向传播计算

        顺序通过每一层，计算最终输出。

        Args:
            x: 输入向量

        Returns:
            Value 或 list: 最后一层的输出
        """
        # 逐层前向传播
        # 每层的输出作为下一层的输入
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        返回所有可学习参数

        Returns:
            list: 所有层的所有参数
        """
        # 展开所有层的参数
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        字符串表示

        Returns:
            str: MLP 结构描述
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# ================ 测试代码 ================

def _test():
    """
    简单测试函数
    """
    # 测试单个神经元
    print("=== Neuron Test ===")
    neuron = Neuron(3)
    print(neuron)  # ReLUNeuron(3)

    x = [1.0, 2.0, 3.0]
    out = neuron(x)
    print(f"Input: {x}")
    print(f"Output: {out}")

    # 测试 Layer
    print("\n=== Layer Test ===")
    layer = Layer(2, 3)
    print(layer)

    x = [1.0, 2.0]
    out = layer(x)
    print(f"Input: {x}")
    print(f"Output: {out}")

    # 测试 MLP
    print("\n=== MLP Test ===")
    mlp = MLP(2, [8, 4, 1])
    print(mlp)
    print(f"Parameters: {len(mlp.parameters())}")

    x = [1.0, 2.0]
    out = mlp(x)
    print(f"Input: {x}")
    print(f"Output: {out}")

    # 测试梯度
    print("\n=== Gradient Test ===")
    mlp = MLP(2, [4, 2])
    x = [1.5, 2.5]
    y_true = Value(1.0)

    # 前向传播
    y_pred = mlp(x)
    if not isinstance(y_pred, Value):
        y_pred = y_pred[0]

    # 计算损失
    loss = (y_pred - y_true) ** 2

    # 反向传播
    loss.backward()

    # 检查梯度
    print(f"Loss: {loss.data}")
    print(f"Number of parameters with gradients: {len([p for p in mlp.parameters() if p.grad != 0])}")


if __name__ == "__main__":
    _test()