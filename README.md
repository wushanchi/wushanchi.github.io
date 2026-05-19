# Neural Networks: Zero to Hero — GPT 学习路线图

> ⭐ 基于 Andrej Karpathy 的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) 构建的中文学习指南
>
> 本仓库包含视频的详细中文注释、代码解读和学习笔记。

---

## 🎯 学习路线图总览

```
阶段一: 神经网络基础 (Lectures 1-2)
    ↓
阶段二: MLP 与现代训练技术 (Lectures 3-6)
    ↓
阶段三: Transformer 核心 (Lectures 7-8) ⭐
    ↓
阶段四: GPT 训练与微调 (Lecture 9+)
    ↓
阶段五: 底层实现与进阶 (可选)
```

---

## 📚 阶段一: 神经网络基础

### Lecture 1: 反向传播 (Backpropagation)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| **代码演示** | [lectures/micrograd](./lectures/micrograd/) |
| **代码仓库** | [micrograd](https://github.com/karpathy/micrograd) |
| **中文注释版** | [micrograd-comment](./micrograd-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/github/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb) |
| **核心内容** | 反向传播算法、梯度下降、PyTorch 基础 |

**课程介绍**:
本课程介绍神经网络反向传播算法（Backpropagation）的实现。我们从零开始构建一个轻量级的 autograd 引擎 `micrograd`，它能在动态计算图上进行反向模式自动微分。通过约 150 行代码，你将理解神经网络训练的核心机制。课程假设具备基本的 Python 知识和对微积分的模糊记忆。

**实现原理**:

```
核心类: Value — 存储标量值 + 梯度

前向传播:
  a = Value(2.0)
  b = Value(3.0)
  c = a * b + a  # 自动构建计算图

反向传播:
  c.backward()  # 链式法则自动计算所有梯度

梯度公式:
  加法: ∂L/∂a = out.grad
  乘法: ∂L/∂a = b.data * out.grad
  ReLU: ∂L/∂a = (a > 0) * out.grad
```

**维度说明**:
- 输入: 标量 [] (0维张量)
- 输出: 标量 [] (0维张量)
- 参数: 权重向量 w[nin] + 偏置 b[]

---

### Lecture 2: 语言建模入门 (Language Modeling)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| **代码演示** | [lectures/makemore/makemore_part1_bigrams.ipynb](./lectures/makemore/) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-comment](./makemore-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/github/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb) |
| **核心内容** | 字符级语言模型、Bigram 模型、PyTorch Tensor |

**课程介绍**:
我们实现一个字符级 Bigram 语言模型，这是后续视频中逐步复杂化为现代 Transformer 语言模型（如 GPT）的基础。本视频重点关注：(1) 介绍 `torch.Tensor` 及其在高效评估神经网络时的细节；(2) 语言建模的总体框架，包括模型训练、采样和损失函数评估（如分类的负对数似然）。

**实现原理**:

```
Bigram 模型 — 最简单的语言模型

架构:
  logits = nn.Parameter(torch.zeros((vocab_size, vocab_size)))
  output = logits[idx]  # 查表操作

维度:
  输入 idx: (B, T) — 字符索引
  输出 logits: (B, T, vocab_size)

训练: 交叉熵损失 = CrossEntropy(logits, targets)
采样: 根据概率分布选择下一个字符
```

**维度变化**:
| 操作 | 输入 | 输出 |
|------|------|------|
| 查表 | (B, T) | (B, T, V) |
| Softmax | (B, T, V) | (B, T, V) 概率 |

---

## 📚 阶段二: MLP 与现代训练技术

### Lecture 3: 多层感知机 (MLP)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/TCH_1BHY58I) |
| **代码演示** | [lectures/makemore/makemore_part2_mlp.ipynb](./lectures/makemore/) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-comment](./makemore-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/github/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) |
| **核心内容** | MLP 架构、过拟合/欠拟合、学习率调优 |

**课程介绍**:
我们实现一个多层感知机（MLP）字符级语言模型。本视频还介绍了机器学习的许多基础知识：模型训练、学习率调优、超参数设置、评估方法、训练/验证/测试集划分、欠拟合与过拟合等。

**实现原理**:

```
MLP (Bengio 2003) — 多层感知机语言模型

架构:
  wte: 嵌入表 (vocab_size, n_embd)
  mlp: Linear(n_embd*block_size, n_embd2) → Tanh → Linear(n_embd2, vocab_size)

前向传播:
  embs = [wte(idx[i]) for i in range(block_size)]  # 收集历史嵌入
  x = cat(embs, dim=-1)  # 拼接
  logits = mlp(x)

维度:
  输入: (B, T) — 字符索引
  嵌入: (B, T, n_embd)
  拼接: (B, T, n_embd*block_size)
  输出: (B, T, vocab_size)
```

---

### Lecture 4: 激活函数与 BatchNorm

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/P6sfmUTpUmc) |
| **代码演示** | [lectures/makemore/makemore_part3_bn.ipynb](./lectures/makemore/) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-comment](./makemore-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/github/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb) |
| **核心内容** | 激活函数统计、BatchNorm 原理、梯度诊断 |

**课程介绍**:
我们深入探讨多层 MLP 的内部机制，审视前向传播激活值的统计量、后向传播梯度的行为，以及当它们缩放不当时的常见陷阱。同时介绍常用的诊断工具和可视化方法，帮助理解深度网络的健康状况。我们还将学习为何训练深度神经网络曾经非常困难，以及首个使这一切变得容易的现代创新：批归一化（Batch Normalization）。

**核心原理**:

```
BatchNorm 公式:
  y = gamma * (x - mean) / sqrt(var + eps) + beta

作用:
  1. 标准化输入，稳定训练
  2. 允许更高的学习率
  3. 提供正则化效果

梯度流:
  forward: x → norm → scale → y
  backward: ∂L/∂y → ∂L/∂gamma, ∂L/∂beta, ∂L/∂x
```

---

### Lecture 5: 手写反向传播 (Backprop Ninja)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/q8SA3rM6ckI) |
| **代码演示** | [lectures/makemore/makemore_part4_backprop.ipynb](./lectures/makemore/) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-comment](./makemore-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing) |
| **核心内容** | 手动实现反向传播、Cross Entropy、LayerNorm |

**课程介绍**:
我们取前一个视频中的 2 层 MLP（带 BatchNorm），手动实现反向传播，不再使用 PyTorch autograd 的 `loss.backward()`。即手动反向传播穿过交叉熵损失、第二个线性层、tanh、批归一化、第一个线性层和嵌入表。在此过程中，你将对梯度如何在计算图中反向流动获得直观的理解，并在高效 Tensors 的层面而非像 micrograd 那样在单个标量上操作。这有助于建立对神经网络优化的能力和直觉，为你在现代神经网络的调试和创新打下基础。

**核心原理**:

```
链式法则:
  ∂L/∂x = ∂L/∂y * ∂y/∂x

操作梯度:
  加法: da = dc, db = dc
  乘法: da = b*dc, db = a*dc
  ReLU: da = (a>0)*dc
  Softmax: da = p - y
```

---

### Lecture 6: WaveNet 与 CNN

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/t3YJ5hKiMQ0) |
| **代码演示** | [lectures/makemore/makemore_part5_cnn1.ipynb](./lectures/makemore/) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-comment](./makemore-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/github/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb) |
| **核心内容** | 卷积神经网络、膨胀卷积、层次化结构 |

**课程介绍**:
我们将前一个视频的 2 层 MLP 改造成树状结构，得到类似于 DeepMind 的 WaveNet (2016) 的卷积神经网络架构。WaveNet 论文中使用因果膨胀卷积（causal dilated convolutions）更高效地实现了相同的层次结构。同时我们更深入地理解 `torch.nn` 是什么、它内部如何工作，以及典型的深度学习开发过程是什么样的（大量阅读文档、跟踪多维张量形状、在 Jupyter notebook 和仓库代码之间切换……）。

**核心原理**:

```
膨胀卷积 (Dilated Convolution):
  膨胀率 d 的卷积核感受野 = k + (k-1)*(d-1)

因果卷积:
  输出[t] 仅依赖于输入[0..t]

WaveNet 堆叠:
  [d=1] → [d=2] → [d=4] → [d=8] → ...
  总感受野 = 2^n - 1
```

---

## 📚 阶段三: Transformer 核心 ⭐

### Lecture 7: Let's build GPT

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| **代码演示** | [lectures/nanogpt](./lectures/nanogpt/) |
| **代码仓库** | [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) / [nanoGPT](https://github.com/karpathy/nanoGPT) |
| **中文注释版** | [nanoGPT-comment](./nanoGPT-comment/) |
| **核心内容** | Transformer 架构、Self-Attention、GPT 实现 |

**课程介绍**:
我们从零开始构建生成式预训练 Transformer（GPT），遵循论文 "Attention is All You Need" 以及 OpenAI 的 GPT-2/3。我们讨论与 ChatGPT 的联系，后者已风靡全球。我们观看 GitHub Copilot（本身也是一个 GPT）如何帮助我们写一个 GPT（元编程！）。建议观众先观看之前的 makemore 视频，熟悉自回归语言建模框架和 PyTorch nn 的基础知识。

**⭐ 这是课程的核心视频！**

**实现原理**:

```
Self-Attention 公式:
  Attention(Q,K,V) = softmax(QK^T / √d_k) V

Multi-Head Attention:
  head_i = Attention(Q_i, K_i, V_i)
  output = concat(head_1, ..., head_n) * W_o

维度:
  输入 x: (B, T, n_embd)
  QKV: (B, T, 3*n_embd)
  分割 → (B, T, n_head, hs)
  转置 → (B, n_head, T, hs)
  注意力: (B, n_head, T, hs)
  拼接 → (B, T, n_embd)
```

**GPT 架构**:
```
Token Embeddings + Position Embeddings
    ↓
[Block × n_layer]
  └─ LayerNorm → Self-Attention → 残差
  └─ LayerNorm → MLP → 残差
    ↓
LayerNorm
    ↓
LM Head (Linear)
    ↓
Output
```

**GPT-2 配置**:
| 参数 | 值 |
|------|-----|
| vocab_size | 50257 |
| block_size | 1024 |
| n_layer | 12 |
| n_head | 12 |
| n_embd | 768 |
| 参数量 | 124M |

---

### Lecture 8: GPT Tokenizer

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) |
| **代码仓库** | [minbpe](https://github.com/karpathy/minbpe) |
| **中文注释版** | [minbpe-comment](./minbpe-comment/) |
| **Colab** | [Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing) |
| **核心内容** | BPE 算法、Tokenization 问题、GPT Tokenizer |

**课程介绍**:
分词器（Tokenizer）是大型语言模型（LLM）中必要且普遍的组件，负责在字符串和 token（文本片段）之间进行翻译。分词器是 LLM 流程中完全独立的一个阶段：它们有自己的训练集、训练算法（字节对编码 BPE），训练后实现两个基本函数：`encode()`（从字符串到 token）和 `decode()`（从 token 到字符串）。本课程从零开始构建 OpenAI GPT 系列使用的分词器。在此过程中，我们将看到许多 LLM 的奇怪行为和问题实际上可以追溯到分词环节。

**核心原理**:

```
BPE 算法:
  1. 将文本分解为字节序列
  2. 统计相邻字节对频率
  3. 合并最频繁的对
  4. 重复直到达到词汇表大小

词汇表构建:
  初始: 256 字节
  合并: 每次添加 1 个新 token
  最终: 50257 tokens (GPT-2)
```

---

## 📚 阶段四: GPT 训练与微调

### Lecture 9: 复现 GPT-2 (124M)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/l8pRSuU81PU) |
| **代码演示** | [nanogpt-build-comment](./nanogpt-build-comment/) |
| **代码仓库** | [build-nanogpt](https://github.com/karpathy/build-nanogpt) / [nanoGPT](https://github.com/karpathy/nanoGPT) |
| **中文注释版** | [nanoGPT-comment](./nanoGPT-comment/) |
| **核心内容** | 从零训练 GPT-2、使用 FineWeb 数据集 |

**课程介绍**:
我们使用 nanoGPT 从零开始训练一个 GPT-2 (124M) 模型。课程涵盖完整的训练流程，包括：使用 FineWeb 数据集进行预训练、实现 AdamW 优化器和权重衰减、设置分布式训练（DDP）、以及使用 HellaSwag 进行评估。在此过程中，你将学习到如何从头构建一个生产级别的语言模型训练 pipeline。

**核心原理**:

```
训练配置:
  batch_size: 8-16
  learning_rate: 1e-3 (AdamW)
  weight_decay: 0.1
  max_steps: 10000+

训练循环:
  1. X, Y = get_batch()
  2. logits, loss = model(X, Y)
  3. loss.backward()
  4. optimizer.step()

监控指标:
  - train loss (应下降)
  - eval loss (应在 500 步后稳定)
  - grad norm (应 < 1.0)
```

---

## 📚 阶段五: 底层实现与进阶 (可选)

### 底层训练: LLM.C

| 项目 | 内容 |
|------|------|
| **代码仓库** | [llm.c](https://github.com/karpathy/llm.c) |
| **中文注释版** | [llm-c-comment](./llm-c-comment/) |
| **核心内容** | 纯 C/CUDA 实现 LLM 训练，零依赖 |

**核心原理**:

```
纯 C 实现的优势:
  - 零依赖（不需要 Python/PyTorch）
  - 极致性能
  - 易于部署

CUDA 优化:
  - 矩阵乘法 kernel
  - 共享内存
  - 合并内存访问
```

### Llama2 推理: llama2.c

| 项目 | 内容 |
|------|------|
| **代码仓库** | [llama2.c](https://github.com/karpathy/llama2.c) |
| **中文注释版** | [llama2-c-comment](./llama2-c-comment/) |
| **核心内容** | 单文件纯 C 推理 Llama 2 |

**核心原理**:

```
与 GPT-2 的区别:
  - 位置编码: RoPE (Rotary)
  - FFN: SwiGLU 激活
  - 归一化: RMSNorm
  - 注意力: MQA (Multi-Query Attention)
```

### BPE 算法: minbpe

| 项目 | 内容 |
|------|------|
| **代码仓库** | [minbpe](https://github.com/karpathy/minbpe) |
| **中文注释版** | [minbpe-comment](./minbpe-comment/) |
| **核心内容** | 最简 BPE 算法实现 |

---

## 🗂️ 仓库结构

```
karpathy-gpt-roadmap/
├── README.md                          # 本文件
├── SETUP.md                           # Python 环境配置指南
├── colabs/                            # Google Colab 笔记本链接
│
├── lectures/                          # 原始课程代码（notebooks）
│   ├── micrograd/                    # Lecture 1-2: 反向传播 Jupyter notebooks
│   │   ├── micrograd_lecture_first_half_roughly.ipynb
│   │   └── micrograd_lecture_second_half_roughly.ipynb
│   ├── makemore/                     # Lecture 2-6: makemore Jupyter notebooks
│   │   ├── makemore_part1_bigrams.ipynb
│   │   ├── makemore_part2_mlp.ipynb
│   │   ├── makemore_part3_bn.ipynb
│   │   ├── makemore_part4_backprop.ipynb
│   │   └── makemore_part5_cnn1.ipynb
│   └── nanogpt/                      # Lecture 9: GPT 代码演示
│       ├── README.md
│       ├── bigram.py
│       ├── gpt.py
│       ├── input.txt
│       └── more.txt
│
├── micrograd-comment/                # Lecture 1 中文注释
│   ├── code/engine.py               # 核心代码 + 中文注释
│   ├── code/nn.py                   # 神经网络模块 + 中文注释
│   ├── notes.md                     # 学习笔记
│   └── index.html                   # 交互式代码解析网页
│
├── makemore-comment/                  # Lecture 2-6 中文注释（汇总）
│   ├── code/                          # 各部分代码
│   │   ├── makemore_part1.py         # Bigram 语言模型
│   │   ├── makemore_part2.py         # MLP 语言模型
│   │   ├── makemore_part3.py         # BatchNorm 与激活函数
│   │   ├── makemore_part4.py         # 手动反向传播
│   │   └── makemore_part5.py         # CNN/WaveNet
│   ├── notes.md                       # 学习笔记汇总
│   └── README.md                     # 课程概览
│
├── nanogpt-build-comment/           # Lecture 7 中文注释（Transformer 核心）
│   └── notes.md                     # Transformer 详细原理
│
├── minbpe-comment/                  # Lecture 8 中文注释（BPE Tokenizer）
│
├── nanogpt-build-comment/           # Lecture 9 代码演示（build-nanogpt）
│
├── nanoGPT-comment/                 # nanoGPT 完整中文注释版
│   ├── model.py                     # GPT 模型定义 + 详细中文注释
│   ├── train.py                     # 训练脚本 + 详细中文注释
│   ├── sample.py                    # 采样脚本 + 详细中文注释
│   ├── bench.py                     # 基准测试 + 详细中文注释
│   ├── configurator.py              # 配置管理 + 详细中文注释
│   ├── config/                      # 配置文件目录
│   └── data/                        # 数据准备脚本目录
│
└── 阶段五/                           # 底层实现 (可选)
    ├── llm-c-comment/
    ├── llama2-c-comment/
    └── minbpe-comment/
```

---

## 📖 学习建议

### 推荐学习顺序

```
第1-2周 → Lecture 1-2 (基础)
第3-4周 → Lecture 3-4 (MLP + BatchNorm)
第5-6周 → Lecture 5-6 (反向传播 + CNN)
第7周   → Lecture 7 ⭐ (GPT 核心 - 必看!)
第8周   → Lecture 8-9 (Tokenizer + 训练)
第9周+  → 实战项目
```

### 学习技巧

1. **边看边敲**: 每个视频都配套 Jupyter Notebook，先跟着敲一遍
2. **做笔记**: 每个 lecture 都有对应的注释文件夹，用于记录学习心得
3. **完成练习**: 视频描述中都有练习题，试着自己做
4. **实践项目**: 完成后尝试用自己的数据集微调模型

---

## 🔗 相关资源链接

### 原始仓库

| 仓库 | 描述 |
|------|------|
| [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 主课程仓库 |
| [micrograd](https://github.com/karpathy/micrograd) | 反向传播实现 |
| [makemore](https://github.com/karpathy/makemore) | 字符级语言模型 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | GPT 训练库（完整版）|
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | GPT 从零构建（视频配套）|
| [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | GPT 简化版演示 |
| [minbpe](https://github.com/karpathy/minbpe) | BPE Tokenizer |
| [llm.c](https://github.com/karpathy/llm.c) | C/CUDA 训练 |
| [llama2.c](https://github.com/karpathy/llama2.c) | 单文件 Llama2 |

### 视频播放列表

- **YouTube**: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

## 📝 贡献指南

如果你想为仓库添中文注释或学习笔记:

1. Fork 本仓库
2. 在对应的 `-comment` 文件夹中添加你的笔记
3. 提交 Pull Request

---

## 📄 License

MIT License - 参见 [LICENSE](./LICENSE)

---

> 本项目仅供学习使用，内容版权归 Andrej Karpathy 所有。
> 视频链接来自 [YouTube](https://www.youtube.com/@AndrejKarpathy)
