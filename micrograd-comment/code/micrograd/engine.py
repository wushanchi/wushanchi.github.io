"""
micrograd 引擎 — 自动求导核心实现

本模块实现了 micrograd 的核心数据结构 Value 类，用于构建计算图和自动求导。

来源: https://github.com/karpathy/micrograd

============ 核心概念 ============

【计算图】
每次运算（+、*、**、ReLU 等）都会创建一个新的 Value 节点，
自动记录前驱节点(_prev)和操作类型(_op)，构建有向无环图(DAG)。

【反向传播】
从输出节点开始，通过链式法则反向计算每个节点的梯度：
- 加法梯度: ∂L/∂a = ∂L/∂out × 1 = out.grad
- 乘法梯度: ∂L/∂a = ∂L/∂out × b = other.data × out.grad
- ReLU梯度: ∂L/∂a = ∂L/∂out × (a > 0) = (out.data > 0) × out.grad

【拓扑排序】
反向传播前需要确定节点的执行顺序，确保从叶节点到根节点依次计算。
使用深度优先搜索(DFS)构建拓扑排序列表。

============ 代码结构 ============

Value 类主要方法:
- __init__: 初始化节点数据
- __add__, __mul__, __pow__: 运算符重载
- relu: ReLU 激活函数
- backward: 反向传播入口
- 辅助运算: __neg__, __sub__, __truediv__ 等
"""

import random


class Value:
    """
    自动求导的基本单元 — 存储标量值和梯度

    每个 Value 对象代表计算图中的一个节点，包含:
    - data: 标量数值
    - grad: 梯度值（默认为0，反向传播时计算）
    - _backward: 反向传播函数
    - _prev: 前驱节点集合（用于构建计算图）
    - _op: 操作类型（用于调试和可视化）
    """

    def __init__(self, data, _children=(), _op=''):
        """
        初始化 Value 对象

        Args:
            data: 标量数值（float 或 int）
            _children: 前驱节点集合（元组），默认为空
            _op: 操作类型字符串，用于调试（默认为空）

        Example:
            a = Value(2.0)           # 创建数值节点
            b = Value(3.0)
            c = Value(a.data + b.data, (a, b), '+')  # 创建加法节点
        """
        self.data = data                # 标量数值，如 2.5
        self.grad = 0                   # 梯度值，初始化为0
        self._backward = lambda: None  # 反向传播函数，默认为空函数
        self._prev = set(_children)     # 前驱节点集合，用于构建计算图
        self._op = _op                  # 操作类型，用于调试（如 '+', '*', 'ReLU'）

    def __add__(self, other):
        """
        加法运算: self + other

        构建计算图节点，定义反向传播函数。

        前向传播: out = a + b
        反向传播: da = out.grad × 1, db = out.grad × 1

        Args:
            other: Value 对象或数值

        Returns:
            Value: 新的 Value 节点

        Example:
            c = a + b      # 自动构建计算图
            c = a + 2      # 自动将 2 转换为 Value(2)
        """
        # 类型标准化：如果 other 不是 Value 对象，则转换为 Value
        other = other if isinstance(other, Value) else Value(other)

        # 创建新的 Value 节点，记录前驱节点和操作类型
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """
            反向传播函数：计算加法对前驱节点的梯度

            加法梯度规则：
            - ∂(a+b)/∂a = 1
            - ∂(a+b)/∂b = 1

            因此：self.grad += out.grad, other.grad += out.grad
            """
            self.grad += out.grad        # ∂L/∂a = ∂L/∂out × ∂out/∂a = out.grad × 1
            other.grad += out.grad       # ∂L/∂b = ∂L/∂out × ∂out/∂b = out.grad × 1

        out._backward = _backward  # 将反向传播函数绑定到输出节点
        return out

    def __mul__(self, other):
        """
        乘法运算: self * other

        前向传播: out = a × b
        反向传播: da = b × out.grad, db = a × out.grad

        Args:
            other: Value 对象或数值

        Returns:
            Value: 新的 Value 节点
        """
        # 类型标准化
        other = other if isinstance(other, Value) else Value(other)

        # 创建新的 Value 节点
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            反向传播函数：计算乘法对前驱节点的梯度

            乘法梯度规则：
            - ∂(a×b)/∂a = b
            - ∂(a×b)/∂b = a

            因此：self.grad += other.data × out.grad
            """
            self.grad += other.data * out.grad    # ∂L/∂a = ∂L/∂out × ∂out/∂a = out.grad × b
            other.grad += self.data * out.grad   # ∂L/∂b = ∂L/∂out × ∂out/∂b = out.grad × a

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        幂运算: self ** other

        前向传播: out = a^n
        反向传播: da = n × a^(n-1) × out.grad

        Args:
            other: 幂指数（int 或 float）

        Returns:
            Value: 新的 Value 节点
        """
        # 目前只支持 int 或 float 类型的幂指数
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        # 创建新的 Value 节点
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            """
            反向传播函数：计算幂运算对底数的梯度

            幂函数梯度规则：
            - ∂(a^n)/∂a = n × a^(n-1)

            因此：self.grad += other × self.data^(other-1) × out.grad
            """
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU（Rectified Linear Unit）激活函数

        前向传播: out = max(0, a)
        反向传播: da = out.grad if a > 0 else 0

        ReLU 的几何意义：
        - 当输入 > 0 时，输出等于输入（梯度为1）
        - 当输入 <= 0 时，输出为0（梯度为0）

        Returns:
            Value: 新的 Value 节点
        """
        # 创建新的 Value 节点
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            """
            反向传播函数：计算 ReLU 对输入的梯度

            ReLU 梯度规则：
            - ∂ReLU/∂a = 1  当 a > 0
            - ∂ReLU/∂a = 0  当 a <= 0

            由于我们用 out.data（即 max(0, a)）判断，
            所以条件为 out.data > 0（等价于 a > 0）
            """
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        反向传播入口函数

        执行两步：
        1. 拓扑排序：从当前节点出发，DFS遍历所有前驱节点，构建执行顺序
        2. 链式法则：逆序遍历拓扑列表，依次调用每个节点的 _backward()

        执行流程示例：
            假设计算图：a → b → c → out
            拓扑排序后：topo = [a, b, c, out]
            反向传播顺序：reversed(topo) = [out, c, b, a]

        注意：
            - 需要手动设置 out.grad = 1（损失函数对自身的梯度）
            - 梯度是累加的（+=），支持多个输出路径
        """
        # ===== 步骤1：拓扑排序 =====
        # 构建从叶节点到当前节点的执行顺序
        topo = []           # 拓扑排序结果
        visited = set()      # 已访问节点集合

        def build_topo(v):
            """
            深度优先搜索构建拓扑排序

            Args:
                v: 当前节点（Value 对象）
            """
            if v not in visited:
                visited.add(v)                    # 标记为已访问
                for child in v._prev:             # 递归访问所有前驱节点
                    build_topo(child)
                topo.append(v)                    # 后序遍历：所有子节点处理完后添加自身

        build_topo(self)  # 从当前节点开始构建拓扑排序

        # ===== 步骤2：逆序应用链式法则 =====
        self.grad = 1  # 初始化当前节点的梯度为1（损失函数对自身的偏导为1）

        # 逆序遍历拓扑列表，依次执行反向传播
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        """
        负号运算: -self

        等价于 self * -1

        Returns:
            Value: 新的 Value 节点
        """
        return self * -1

    def __radd__(self, other):  # other + self
        """
        反向加法: other + self

        当左操作数不支持加法时调用（如 int + Value）

        Args:
            other: 左操作数

        Returns:
            Value: 新的 Value 节点
        """
        return self + other

    def __sub__(self, other):  # self - other
        """
        减法运算: self - other

        实现为 self + (-other)

        Args:
            other: Value 对象或数值

        Returns:
            Value: 新的 Value 节点
        """
        return self + (-other)

    def __rsub__(self, other):  # other - self
        """
        反向减法: other - self

        当左操作数不支持减法时调用

        Args:
            other: 左操作数

        Returns:
            Value: 新的 Value 节点
        """
        return other + (-self)

    def __rmul__(self, other):  # other * self
        """
        反向乘法: other * self

        当左操作数不支持乘法时调用（如 int * Value）

        Args:
            other: 左操作数

        Returns:
            Value: 新的 Value 节点
        """
        return self * other

    def __truediv__(self, other):  # self / other
        """
        除法运算: self / other

        实现为 self * other^(-1)

        Args:
            other: Value 对象或数值

        Returns:
            Value: 新的 Value 节点
        """
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        """
        反向除法: other / self

        当左操作数不支持除法时调用

        Args:
            other: 左操作数

        Returns:
            Value: 新的 Value 节点
        """
        return other * self ** -1

    def __repr__(self):
        """
        字符串表示，用于调试

        Returns:
            str: Value 对象的字符串表示
        """
        return f"Value(data={self.data}, grad={self.grad})"


def _test():
    """
    简单的功能测试
    """
    # 测试基本运算
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a ** 2
    c.backward()

    print(f"a = {a.data}, a.grad = {a.grad}")  # da = b + 2a = 3 + 4 = 7
    print(f"b = {b.data}, b.grad = {b.grad}")  # db = a = 2
    print(f"c = {c.data}, c.grad = {c.grad}")   # dc = 1

    # 测试 ReLU
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    print(f"x = {x.data}, x.grad = {x.grad}")  # 负数区间，梯度为0

    x2 = Value(2.0)
    y2 = x2.relu()
    y2.backward()
    print(f"x2 = {x2.data}, x2.grad = {x2.grad}")  # 正数区间，梯度为1


if __name__ == "__main__":
    _test()