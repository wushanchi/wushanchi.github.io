# Makemore 学习笔记汇总

本文件夹包含 makemore 系列的完整学习笔记，按 5 个部分组织。

---

## Part 1: Bigram 语言模型 (Lecture 2)

### 核心概念

**语言建模**：给定前文，预测下一个词的概率分布 P(下一个词 | 前文)

**Bigram 模型**：最简单的 baseline，只根据前一个字符预测下一个字符
```
P("hello") = P("h"|"<START>") × P("e"|"h") × P("l"|"e") × P("l"|"e") × P("o"|"l")
```

### 关键代码

```python
class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def forward(self, idx):
        logits = self.logits[idx]  # (B, T, vocab_size)
        return logits
```

### 维度变化

| 操作 | 输入 | 输出 |
|------|------|------|
| 查表 | (B, T) | (B, T, V) |
| Softmax | (B, T, V) | (B, T, V) 概率 |

---

## Part 2: MLP 语言模型 (Lecture 3)

### 核心概念

**MLP 架构（Bengio 2003）**：
1. 将前 block_size 个字符的嵌入拼接
2. 通过 MLP 计算下一个字符的 logits

### 维度变化

```
idx: (B, T)
  ↓
wte(idx): (B, T, n_embd)
  ↓
拼接: (B, T, n_embd * block_size)
  ↓
MLP: (B, T, vocab_size)
```

### 过拟合与欠拟合

| 问题 | 解决方法 |
|------|----------|
| 过拟合 | 增加正则化、减小模型、增加数据 |
| 欠拟合 | 增大模型、训练更久、调整学习率 |

---

## Part 3: BatchNorm 与激活函数 (Lecture 4)

### BatchNorm 公式

```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

### 为什么有效？

1. **缓解内部协变量偏移**：每层输入分布稳定
2. **允许更高的学习率**：梯度更稳定
3. **有一定的正则化效果**：batch 噪声

### 梯度诊断

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| 梯度消失 | 层太深、激活函数饱和 | 残差连接、合适初始化 |
| 梯度爆炸 | 学习率太高、权重太大 | 梯度裁剪、减小学习率 |
| NaN 损失 | 除零、log(0) | 添加 eps、检查数据 |

---

## Part 4: 手写反向传播 (Lecture 5)

### 链式法则

```
∂L/∂x = ∂L/∂y * ∂y/∂x
```

### 常见操作的梯度

| 操作 | forward | backward |
|------|---------|----------|
| 加法 | c = a + b | da = dc, db = dc |
| 乘法 | c = a * b | da = b * dc, db = a * dc |
| ReLU | c = max(0, a) | da = dc if a > 0 else 0 |
| Softmax | p = exp(a)/Σexp(a) | da = p - y |

### LayerNorm 公式

```
μ = mean(x)
σ² = var(x)
x_norm = (x - μ) / sqrt(σ² + eps)
y = gamma * x_norm + beta
```

---

## Part 5: WaveNet 与 CNN (Lecture 6)

### 膨胀卷积 (Dilated Convolution)

普通卷积：膨胀率 = 1，感受野 = k
膨胀卷积：膨胀率 = d，感受野 = k + (k-1)*(d-1)

### WaveNet 堆叠

```
层 1: 膨胀率 1
层 2: 膨胀率 2
层 4: 膨胀率 4
层 8: 膨胀率 8
...
总感受野 = 2^n - 1
```

### 因果卷积约束

输出[t] 只能依赖输入[0..t]

---

## 模型演进图

```
Bigram (Part 1)
    ↓
MLP (Part 2)
    ↓
MLP + BatchNorm (Part 3)
    ↓
手动反向传播 (Part 4)
    ↓
CNN / WaveNet (Part 5)
    ↓
Transformer / GPT (Lecture 7)
```

---

> 📚 视频: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)
