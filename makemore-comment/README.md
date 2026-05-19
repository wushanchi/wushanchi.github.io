# Makemore — 字符级语言模型系列

> 本系列包含 5 个 Jupyter Notebooks，讲解从简单 Bigram 到复杂 CNN/WaveNet 的字符级语言模型实现。

## 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | 字符级语言模型、Bigram、MLP、BatchNorm、反向传播、CNN/WaveNet |

---

## 📚 各部分内容

### Part 1: Bigram 语言模型 (Lecture 2)

**代码演示**: [makemore_part1_bigrams.ipynb](./code/makemore_part1.py)

学习内容：
- 什么是语言建模
- PyTorch Tensor 基础操作
- Bigram 模型实现
- 训练/采样/评估流程
- 困惑度（Perplexity）指标

### Part 2: MLP 语言模型 (Lecture 3)

**代码演示**: [makemore_part2_mlp.ipynb](./code/makemore_part2.py)

学习内容：
- MLP 架构（Bengio 2003）
- 嵌入层和拼接操作
- 过拟合与欠拟合诊断
- 学习率调优
- 训练/验证/测试集划分

### Part 3: BatchNorm 与激活函数 (Lecture 4)

**代码演示**: [makemore_part3_bn.ipynb](./code/makemore_part3.py)

学习内容：
- 激活值统计与诊断
- BatchNorm 原理与实现
- 梯度流分析
- 权重初始化方法（Xavier/Kaiming）
- 训练稳定性检查

### Part 4: 手写反向传播 (Lecture 5)

**代码演示**: [makemore_part4_backprop.ipynb](./code/makemore_part4.py)

学习内容：
- 链式法则详解
- 常见操作的梯度计算（加法、乘法、ReLU、Softmax）
- Cross Entropy 反向传播
- LayerNorm 手动实现

### Part 5: WaveNet 与 CNN (Lecture 6)

**代码演示**: [makemore_part5_cnn1.ipynb](./code/makemore_part5.py)

学习内容：
- 卷积神经网络基础
- 膨胀卷积（Dilated Convolution）
- 因果卷积（Causal Convolution）
- WaveNet 架构
- 层次化特征提取

---

## 🧠 模型演进

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

## 📊 各模型对比

| 模型 | 参数量 | 复杂度 | 效果 |
|------|--------|--------|------|
| Bigram | ~1K | 最简单 | baseline |
| MLP | ~100K | 中等 | 明显更好 |
| MLP + BatchNorm | ~100K | 中等 | 更稳定 |
| CNN / WaveNet | ~300K | 较高 | 更好 |
| **Transformer** | ~124M | 高 | **最佳** |

---

## 🔗 相关资源

- **视频播放列表**: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **代码仓库**: [karpathy/makemore](https://github.com/karpathy/makemore)
- **nanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — GPT 完整实现

---

> 本课程版权归 Andrej Karpathy 所有。
> 视频链接来自 [YouTube](https://www.youtube.com/@AndrejKarpathy)