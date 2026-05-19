"""
Makemore Part 4 — 手写反向传播 (Backprop Ninja)

本代码演示如何手动实现反向传播，不使用 autograd。

来源: https://github.com/karpathy/makemore
"""

import math


# ============================================================================
# 第一部分: 基础自动求导引擎
# ============================================================================

class Value:
    """
    简化的自动求导 Value 类

    支持基本运算的 forward 和 backward
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """加法: a + b"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 加法梯度: 上游梯度直接传递给所有前驱
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """乘法: a * b"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 乘法梯度: 另一因子 × 上游梯度
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """幂运算: a ** n"""
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """ReLU: max(0, x)"""
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """反向传播"""
        # 拓扑排序
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 逆序执行
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1


# ============================================================================
# 第二部分: softmax 和 Cross Entropy
# ============================================================================

def softmax(logits):
    """
    Softmax 函数

    softmax_i = exp(logits_i) / sum(exp(logits_j))

    Args:
        logits: [logit1, logit2, ..., logit_n]

    Returns:
        probs: [p1, p2, ..., pn] 和为 1
    """
    max_logit = max(logits)  # 数值稳定
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def cross_entropy(logits, target_idx):
    """
    Cross Entropy 损失

    CE = -log(p_target)

    梯度 (当 target 是 one-hot):
    ∂CE/∂logit_i = p_i - y_i
    其中 y_i = 1 if i == target else 0

    Args:
        logits: 未归一化的分数
        target_idx: 目标类别的索引

    Returns:
        loss: 交叉熵损失
    """
    probs = softmax(logits)
    # 取目标类别的概率
    p_target = probs[target_idx]
    # 避免 log(0)
    loss = -math.log(max(p_target, 1e-10))
    return loss


def cross_entropy_grad(logits, target_idx):
    """
    Cross Entropy 的梯度

    ∂CE/∂logit_i = p_i - y_i

    Args:
        logits: 未归一化的分数
        target_idx: 目标类别的索引

    Returns:
        grads: 每个 logit 的梯度
    """
    probs = softmax(logits)
    grads = []

    for i, p in enumerate(probs):
        y = 1 if i == target_idx else 0
        grads.append(p - y)

    return grads


# ============================================================================
# 第三部分: LayerNorm
# ============================================================================

class LayerNorm:
    """
    Layer Normalization

    公式: y = gamma * (x - mean) / sqrt(var + eps) + beta

    与 BatchNorm 的区别:
    - BatchNorm: 在 batch 维度上归一化
    - LayerNorm: 在特征维度上归一化
    """

    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 可学习参数
        self.gamma = [1.0] * normalized_shape
        self.beta = [0.0] * normalized_shape

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入向量 (normalized_shape,)

        Returns:
            y: 标准化后的输出
        """
        # 计算均值
        mean = sum(x) / len(x)

        # 计算方差
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        # 标准化
        x_norm = [(xi - mean) / math.sqrt(var + self.eps) for xi in x]

        # 缩放和平移
        y = [self.gamma[i] * x_norm[i] + self.beta[i] for i in range(len(x))]

        return y

    def backward(self, grad_output):
        """
        反向传播

        计算:
        - ∂L/∂gamma
        - ∂L/∂beta
        - ∂L/∂x
        """
        # 这里简化处理，实际实现更复杂
        grad_gamma = grad_output
        grad_beta = grad_output
        grad_x = grad_output  # 简化

        return grad_gamma, grad_beta, grad_x


# ============================================================================
# 第四部分: 示例计算
# ============================================================================

def manual_backprop_example():
    """
    手动反向传播示例

    计算: loss = (a * b + c) ** 2
    其中 a = 2, b = 3, c = 1

    前向:
    1. a * b = 6
    2. a * b + c = 7
    3. (a * b + c) ** 2 = 49

    反向:
    1. ∂L/∂L = 1
    2. ∂L/∂(a*b+c) = 2 * (a*b+c) = 14
    3. ∂L/∂(a*b) = 1 * 14 = 14
    4. ∂L/∂c = 14
    5. ∂L/∂a = b * 14 = 3 * 14 = 42
    6. ∂L/∂b = a * 14 = 2 * 14 = 28
    """

    print("=== 手动反向传播示例 ===")
    print()

    # 创建变量
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    # 前向传播
    ab = a * b        # 6
    sum_ab_c = ab + c  # 7
    loss = sum_ab_c ** 2  # 49

    print(f"前向传播:")
    print(f"  a = {a.data}")
    print(f"  b = {b.data}")
    print(f"  c = {c.data}")
    print(f"  a * b = {ab.data}")
    print(f"  a * b + c = {sum_ab_c.data}")
    print(f"  loss = {loss.data}")
    print()

    # 反向传播
    loss.backward()

    print(f"反向传播 (梯度):")
    print(f"  ∂L/∂a = {a.grad}")
    print(f"  ∂L/∂b = {b.grad}")
    print(f"  ∂L/∂c = {c.grad}")
    print(f"  ∂L/∂(a*b) = {ab.grad}")
    print()

    # 验证
    print(f"验证 (解析解):")
    print(f"  ∂L/∂a = 2*b*(a*b+c) = 2*3*7 = 42 ✓" if a.grad == 42 else f"  ∂L/∂a 错误")
    print(f"  ∂L/∂b = 2*a*(a*b+c) = 2*2*7 = 28 ✓" if b.grad == 28 else f"  ∂L/∂b 错误")
    print(f"  ∂L/∂c = 2*(a*b+c) = 2*7 = 14 ✓" if c.grad == 14 else f"  ∂L/∂c 错误")


def softmax_example():
    """Softmax 示例"""
    print()
    print("=== Softmax 示例 ===")
    print()

    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)

    print(f"logits: {logits}")
    print(f"softmax: {[f'{p:.4f}' for p in probs]}")
    print(f"sum: {sum(probs):.4f}")
    print()

    # Cross Entropy
    target = 0
    loss = cross_entropy(logits, target)

    print(f"target class: {target}")
    print(f"CE loss: {loss:.4f}")
    print(f"p(target): {probs[target]:.4f}")
    print()

    # 梯度
    grads = cross_entropy_grad(logits, target)
    print(f"梯度 ∂L/∂logit: {[f'{g:.4f}' for g in grads]}")
    print(f"验证: sum(grads) = {sum(grads):.4f} (应该为 0)")


if __name__ == '__main__':
    manual_backprop_example()
    softmax_example()