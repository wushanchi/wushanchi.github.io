# Lecture 1: micrograd — 反向传播与自动微分

本仓库包含 Andrej Karpathy 的 micrograd 课程详细中文注释和学习笔记。

---

## 📚 课程信息

- **视频**: [YouTube - Backpropagation micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- **源代码**: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- **课程难度**: ★★★☆☆ （需要 Python 基础）
- **预计学习时间**: 4-6 小时

---

## 🎯 学习目标

1. 理解**反向传播**的数学原理
2. 掌握**计算图**的构建方法
3. 理解**链式法则**在梯度计算中的应用
4. 实现一个精简的**自动微分引擎**

---

## 📖 课程内容

### Part 1: 自动求导基础

- 计算图的概念
- 前向传播与反向传播
- 链式法则
- 梯度计算规则

### Part 2: Value 类实现

- 数据结构设计
- 运算符重载
- 反向传播函数
- 拓扑排序

### Part 3: 神经网络模块

- Module 基类
- Neuron 单神经元
- Layer 全连接层
- MLP 多层感知机

### Part 4: 训练流程

- 前向传播
- 损失计算
- 反向传播
- 参数更新

---

## 🧠 核心概念详解

### 1. 什么是反向传播？

反向传播（Backpropagation）是训练神经网络的核心算法，通过链式法则从输出层向输入层传播梯度。

```
计算: y = f(g(x))
链式法则: dy/dx = dy/df × df/dg × dg/dx
```

### 2. 梯度计算规则

```
加法: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
     反向传播: self.grad += out.grad, other.grad += out.grad

乘法: ∂(a×b)/∂a = b, ∂(a×b)/∂b = a
     反向传播: self.grad += other.data × out.grad

ReLU: ∂ReLU/∂a = 1 if a > 0 else 0
     反向传播: self.grad += (out.data > 0) × out.grad
```

### 3. 计算图构建

每次进行运算时，micrograd 自动：

1. 创建新的 Value 节点
2. 记录前驱节点（`_prev`）
3. 记录操作类型（`_op`）
4. 定义反向传播函数（`_backward`）

---

## 📁 代码文件说明

| 文件 | 说明 |
|------|------|
| `code/micrograd/engine.py` | 自动求导引擎核心，Value 类实现 |
| `code/micrograd/nn.py` | 神经网络层：Module, Neuron, Layer, MLP |
| `code/micrograd/__init__.py` | 包初始化文件 |
| `code/test/test_engine.py` | 单元测试，对比 PyTorch 验证正确性 |
| `code/demo.ipynb` | Jupyter Notebook 演示 |
| `code/trace_graph.ipynb` | 计算图追踪演示 |

---

## 🔧 运行代码

### 环境准备

```bash
cd micrograd-comment/code
pip install torch  # 仅用于测试验证
```

### 运行测试

```bash
python test/test_engine.py
```

### 交互示例

```python
from micrograd.engine import Value

# 创建变量
x = Value(2.0)
y = Value(3.0)

# 构建计算图
z = x * y + x ** 2

# 反向传播
z.backward()

# 查看梯度
print(f"dz/dx = {x.grad}")  # dz/dx = y + 2x = 3 + 4 = 7
print(f"dz/dy = {y.grad}")  # dz/dy = x = 2
```

---

## 📊 训练示例

```python
from micrograd.nn import MLP

# 创建模型
model = MLP(2, [16, 8, 1])

# 准备数据
x = [1.5, 2.5]
y_true = 1.0

# 前向传播
y_pred = model(x)
if not isinstance(y_pred, Value):
    y_pred = y_pred[0]  # MLP 返回列表时取第一个元素

# 计算损失
loss = (y_pred - Value(y_true)) ** 2

# 反向传播
loss.backward()

# 参数更新
lr = 0.1
for p in model.parameters():
    p.data -= lr * p.grad

# 清除梯度
model.zero_grad()
```

---

## 🤔 常见问题

### Q: micrograd 为什么只处理标量？

A: 为了简化实现。标量运算使计算图更清晰，梯度计算更直观。

### Q: 如何调试计算图？

A: 使用 `_op` 属性查看操作类型，使用 `_prev` 查看前驱节点。

### Q: micrograd 与 PyTorch 的区别？

A: micrograd 是教学目的的最小实现，PyTorch 是生产级框架。核心原理相同。

---

## 🔗 相关资源

- [Karpathy YouTube 频道](https://www.youtube.com/@AndrejKarpathy)
- [minGPT - Karpathy 的 GPT 实现](https://github.com/karpathy/minGPT)
- [makemore - 字符级语言模型](https://github.com/karpathy/makemore)

---

> 📝 本笔记由中文注释版贡献者编写
> ⚠️ 如发现错误欢迎提交 Issue 或 PR
> 📄 许可证与原仓库一致（MIT）