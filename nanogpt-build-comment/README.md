# Lecture 9: 复现 GPT-2 (124M)

> 本课程从零训练 GPT-2 模型，使用 FineWeb 数据集，学习完整的模型训练流程。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) |
| **源代码** | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| **核心主题** | GPT-2 训练、FineWeb 数据集、训练监控、AdamW 优化器 |

---

## 🎯 学习目标

1. 理解 GPT-2 的完整训练流程
2. 掌握数据集处理方法（FineWeb 数据集）
3. 学会监控训练过程（损失、学习率、梯度范数）
4. 理解分布式训练（DDP）的基本原理
5. 掌握 HellaSwag 评估方法

---

## 📁 文件结构

```
nanogpt-build-comment/
├── README.md                      # 项目说明文档
├── notes.md                       # 学习笔记
├── index.html                     # 交互式网页说明
└── code/                          # 代码文件夹（与 karpathy/build-nanogpt 一致）
    ├── train_gpt2.py             # 主要训练脚本（带详细中文注释）
    ├── fineweb.py                # FineWeb 数据集下载与预处理
    ├── hellaswag.py              # HellaSwag 评估数据集
    ├── input.txt                 # 示例输入文件
    ├── play.ipynb               # Jupyter Notebook 交互版
    └── README.md                 # 官方 README
```

---

## 🧠 模型架构

### GPT-2 参数配置

| 参数 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|------|-------------|--------------|-------------|----------|
| n_layer | 12 | 24 | 36 | 48 |
| n_head | 12 | 16 | 20 | 25 |
| n_embd | 768 | 1024 | 1280 | 1600 |
| 参数量 | 124M | 350M | 774M | 1558M |
| vocab_size | 50257 | 50257 | 50257 | 50257 |
| block_size | 1024 | 1024 | 1024 | 1024 |

### 模型组件

```
GPT 模型
├── Token Embeddings (wte): vocab_size → n_embd
├── Position Embeddings (wpe): block_size → n_embd
├── Transformer Blocks (h): n_layer 次
│   └── Block:
│       ├── LayerNorm
│       ├── CausalSelfAttention (Multi-Head)
│       │   ├── QKV 投影: n_embd → 3*n_embd
│       │   └── 输出投影: n_embd → n_embd
│       ├── LayerNorm
│       └── MLP (SwiGLU 激活)
│           ├── 扩展: n_embd → 4*n_embd
│           └── 收缩: 4*n_embd → n_embd
├── Final LayerNorm (ln_f)
└── LM Head: n_embd → vocab_size
```

### 权重共享

GPT-2 使用权重共享：token embeddings (wte) 和 LM head 使用相同的权重矩阵。

这减少了参数量，同时让模型在预测和输入阶段使用一致的表示。

---

## 🔄 训练流程

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| total_batch_size | 524288 (2^19) | 总 batch size（以 token 计） |
| B (micro batch) | 64 | 每个 GPU 的样本数 |
| T (sequence length) | 1024 | 序列长度 |
| grad_accum_steps | 计算得出 | 梯度累积步数 |
| learning_rate | 6e-4 | 最大学习率 |
| min_lr | 6e-5 | 最小学习率（cosine decay 终点） |
| warmup_steps | 715 | Warmup 步数 |
| max_steps | 19073 | 最大训练步数（约 1 epoch） |
| weight_decay | 0.1 | 权重衰减 |

### 学习率调度

```python
def get_lr(it):
    # 1. 线性 warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2. Cosine decay 到 min_lr
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

### 训练循环

```python
for step in range(max_steps):
    # 1. 验证（每 250 步）
    if step % 250 == 0:
        val_loss = evaluate(model)
        save_checkpoint_if_needed()

    # 2. HellaSwag 评估（每 250 步）
    if step % 250 == 0:
        acc = evaluate_hellaswag(model)

    # 3. 文本生成（每 250 步，非第 0 步）
    if step > 0 and step % 250 == 0:
        generated_text = sample(model)

    # 4. 训练步骤
    model.train()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()

    # 5. 梯度裁剪 + 参数更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

---

## 📊 训练监控

### TensorBoard 使用

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/')

# 记录损失
writer.add_scalar("Loss/train", loss.item(), step)
writer.add_scalar("Loss/val", val_loss.item(), step)

# 记录学习率
writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)

# 记录梯度范数
grad_norm = torch.nn.utils.get_grad_norm(model.parameters())
writer.add_scalar("Grad/norm", grad_norm, step)

# 记录 HellaSwag 准确率
writer.add_scalar("Eval/HellaSwag_acc", acc, step)
```

### 监控指标

| 指标 | 说明 | 期望范围 |
|------|------|---------|
| loss | 训练损失 | 下降趋势 |
| eval loss | 验证损失 | 下降趋势，500 步后稳定 |
| grad norm | 梯度范数 | < 1.0 |
| lr | 学习率 | 按照调度器变化 |
| HellaSwag acc | 标准化准确率 | 逐渐提高 |

### 正常训练的特征

- 损失稳定下降
- 梯度范数在 0.5-1.5 之间波动
- 生成文本越来越"有意义"
- HellaSwag 准确率稳步提升

---

## 🧪 数据集处理

### FineWeb 数据集

FineWeb-Edu 是一个高质量教育文本数据集，约 100 亿个 token。

```python
# 下载和预处理
python fineweb.py
# 生成 edu_fineweb10B/ 目录，包含 100 个分片

# 每个分片约 100M tokens
# 第一个分片作为验证集
# 其余作为训练集
```

### 数据加载

```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        # 从分片文件加载数据
        # 支持分布式训练（不同进程读取不同位置）
        pass

    def next_batch(self):
        # 返回 (B, T) 的 x 和 y
        # y 是 x 的下一个 token（自回归目标）
        pass
```

---

## 🏆 HellaSwag 评估

### 评估方法

HellaSwag 是一个常识推理数据集，包含上下文和 4 个候选结尾。

```python
def get_most_likely_row(tokens, mask, logits):
    # 计算每个候选结尾的平均损失
    shift_logits = logits[..., :-1, :]
    shift_tokens = tokens[..., 1:]
    shift_losses = F.cross_entropy(shift_logits, shift_tokens, reduction='none')
    shift_losses = shift_losses.view(4, -1)

    # 只对候选结尾部分（mask=1）计算损失
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)

    # 返回损失最低的候选
    return avg_loss.argmin().item()
```

### 预期性能

| 模型 | acc_norm |
|------|----------|
| 从头训练（随机初始化） | ~25% |
| GPT-2 (124M) | ~31% |
| GPT-2 Medium (350M) | ~38% |
| GPT-2 Large (774M) | ~45% |
| GPT-2 XL (1558M) | ~49% |

---

## 🚀 运行示例

```bash
cd nanogpt-build-comment/code

# 1. 下载 FineWeb 数据集（需要先安装 datasets）
python fineweb.py

# 2. 训练 GPT-2（单 GPU）
python train_gpt2.py

# 3. 训练 GPT-2（多 GPU 分布式）
torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 4. 从预训练 GPT-2 加载权重
# 在 train_gpt2.py 中取消注释:
# model = GPT.from_pretrained("gpt2")
```

---

## 🔧 关键技术点

### 1. 混合精度训练 (bfloat16)

```python
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

### 2. 梯度累积

当 GPU 内存不足以容纳大 batch 时，通过多次小 batch 累加梯度：

```python
loss = loss / grad_accum_steps  # 缩放损失
loss.backward()                  # 梯度累加到 .grad
```

### 3. 分布式数据并行 (DDP)

多 GPU 训练时，每个 GPU 持有完整模型副本，分别计算不同 batch 的梯度，然后同步：

```python
model = DDP(model, device_ids=[ddp_local_rank])
dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
```

### 4. Flash Attention

使用 PyTorch 的 `scaled_dot_product_attention`，自动应用因果掩码且高效：

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 📚 相关资源

- [视频: 复现 GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [代码: karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HellaSwag 数据集](https://github.com/rowanz/hellaswag)
- [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

> 原课程版权归 Andrej Karpathy 所有