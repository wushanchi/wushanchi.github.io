# micrograd — 反向传播与自动微分

> micrograd 是 Andrej Karpathy 实现的一个精简自动微分引擎，是理解 PyTorch 反向传播机制的绝佳入门教材。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| **源代码** | [karpathy/micrograd](https://github.com/karpathy/micrograd) |
| **课程主题** | 自动求导、反向传播、计算图、链式法则 |

---

## 🎯 学习目标

1. 理解反向传播的数学原理（链式法则）
2. 掌握计算图的构建方法
3. 实现一个精简的自动微分引擎
4. 理解神经网络训练的基本流程

---

## 📁 文件结构

```
micrograd-comment/
├── README.md                      # 项目说明文档
├── notes.md                       # 学习笔记
├── index.html                     # 交互式网页说明
└── code/                          # 代码文件夹（与 karpathy/micrograd 结构一致）
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── demo.ipynb                 # Jupyter Notebook 演示
    ├── gout.svg                   # 计算图可视化
    ├── moon_mlp.png               # MLP 可视化图片
    ├── puppy.jpg                  # 示例图片
    ├── setup.py                   # 安装配置
    ├── trace_graph.ipynb          # 计算图追踪
    ├── micrograd/                 # 核心代码包
    │   ├── __init__.py            # 包初始化文件
    │   ├── engine.py              # 自动求导引擎（核心）
    │   └── nn.py                  # 神经网络层定义
    └── test/
        └── test_engine.py        # 单元测试
```

---

## 🧠 核心概念

### 1. 什么是自动微分？

自动微分（Automatic Differentiation）是一种精确计算导数的方法，通过追踪计算过程而不是使用数值近似或符号计算。

```
数值微分（不精确）: (f(x+ε) - f(x)) / ε  →  近似值
符号微分（复杂）:   d/dx[f(x)g(x)]        →  表达式膨胀
自动微分（精确）:   追踪计算图 + 链式法则  →  精确梯度
```

### 2. 计算图

计算图是一种有向无环图（DAG），节点表示变量，边表示操作。

```
示例: y = (a + b) * c

计算图:
    a ──┐
        ├──➕──> d ──┐
    b ──┘           │
                    ├──✕──> y
    c ──────────────┘

前向传播: a=2, b=3, c=4 → d=5 → y=20
反向传播: ∂y/∂a = c = 4, ∂y/∂b = c = 4, ∂y/∂c = d = 5
```

---

## 🔥 engine.py — 自动求导引擎

### Value 类

Value 是 micrograd 的核心，每个 Value 对象存储：

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | float | 标量数值 |
| `grad` | float | 梯度值（反向传播时计算） |
| `_backward` | function | 反向传播函数 |
| `_prev` | set | 前驱节点集合 |
| `_op` | str | 操作类型（用于调试） |

### 支持的运算

| 运算 | 方法 | 梯度公式 |
|------|------|---------|
| 加法 | `__add__` | ∂L/∂a = out.grad |
| 乘法 | `__mul__` | ∂L/∂a = b.data × out.grad |
| 幂运算 | `__pow__` | ∂L/∂a = n × a^(n-1) × out.grad |
| ReLU | `relu()` | ∂L/∂a = (a > 0) × out.grad |

---

## 🏗️ nn.py — 神经网络模块

### Module 基类

所有神经网络层都继承自 Module：

```python
class Module:
    def zero_grad(self):    # 清除所有参数梯度
    def parameters(self):   # 返回可学习参数列表
```

### 层级结构

```
Module (基类)
    │
    ├── Neuron(nin)      # 单神经元: y = ReLU(w·x + b)
    │       │
    │       └── 参数: nin 个权重 + 1 个偏置
    │
    ├── Layer(nin, nout)  # 全连接层: nout 个神经元并行
    │       │
    │       └── 参数: nin×nout 个权重 + nout 个偏置
    │
    └── MLP(nin, nouts)   # 多层感知机: 多个 Layer 堆叠
            │
            └── 参数: 所有 Layer 参数之和
```

### 维度变换

| 模块 | 输入 | 输出 | 参数数量 |
|------|------|------|---------|
| Neuron | [nin] | [] | nin + 1 |
| Layer | [nin] | [nout] | nin×nout + nout |
| MLP | [nin] | [nouts[-1]] | Σ(ninᵢ×ninᵢ₊₁ + ninᵢ₊₁) |

---

## 📊 训练流程

```python
# 1. 构建模型
model = MLP(2, [8, 4, 1])

# 2. 前向传播
x = [1.0, 2.0]
y_pred = model(x)  # 返回 Value 对象

# 3. 计算损失
y_true = Value(1.0)
loss = (y_pred - y_true) ** 2  # MSE 损失

# 4. 反向传播
loss.backward()  # 计算所有参数的梯度

# 5. 更新参数
lr = 0.1
for p in model.parameters():
    p.data -= lr * p.grad

# 6. 清除梯度（下一轮训练）
model.zero_grad()
```

---

## 🧪 运行测试

```bash
cd micrograd-comment/code
python -m pytest test/test_engine.py
# 或直接运行
python test/test_engine.py
```

---

## 🔑 关键设计思想

1. **动态计算图**：每次前向传播实时构建计算图
2. **运算符重载**：通过 Python 特殊方法实现直觉的数学语法
3. **拓扑排序**：保证反向传播按正确顺序执行
4. **链式法则**：精确计算每个参数的梯度

---

## 📚 扩展阅读

- [PyTorch autograd 原理](https://pytorch.org/docs/stable/autograd.html)
- [CS231n 反向传播详解](http://cs231n.stanford.edu/handouts/derivatives.pdf)
- [Chain Rule - 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

---

> 📚 视频: [Backpropagation micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
> 📦 代码: [karpathy/micrograd](https://github.com/karpathy/micrograd)
> 🧪 测试: [test/test_engine.py](https://github.com/karpathy/micrograd/blob/master/test/test_engine.py)