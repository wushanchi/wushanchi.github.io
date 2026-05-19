# 复现 GPT-2 — 完整训练指南

> 本笔记详细介绍从零训练 GPT-2 的完整流程，包括数据处理、模型实现、训练监控和评估。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) |
| **代码仓库** | [build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| **核心主题** | GPT-2 训练、FineWeb 数据集、DDP 分布式训练、HellaSwag 评估 |

---

## 🎯 学习目标

1. 理解 GPT-2 完整架构和参数配置
2. 掌握从零训练 GPT-2 的流程
3. 学会使用 FineWeb 数据集
4. 理解分布式数据并行 (DDP) 训练
5. 掌握 HellaSwag 等评估方法

---

## 🧠 GPT-2 模型架构

### 核心组件

```
GPT-2 (124M)
├── Token Embedding Table: 50257 × 768
├── Position Embedding Table: 1024 × 768
├── Transformer Blocks (×12)
│   ├── LayerNorm
│   ├── Causal Self-Attention
│   │   ├── QKV Linear: 768 → 2304 (3 × 768)
│   │   └── Output Linear: 768 → 768
│   ├── LayerNorm
│   └── MLP
│       ├── Expand Linear: 768 → 3072 (4 × 768)
│       ├── GELU Activation
│       └── Contract Linear: 3072 → 768
├── Final LayerNorm: 768
└── LM Head: 768 → 50257
```

### 关键设计

1. **因果注意力 (Causal Attention)**: 确保 t 位置只能看到 0..t 的 token
2. **残差连接**: 每个子层有残差连接，帮助梯度流
3. **权重共享**: Token embedding 和 LM head 共享权重

### GPT-2 变体对比

| 变体 | 层数 | 头数 | 嵌入维度 | 参数量 |
|------|------|------|----------|--------|
| GPT-2 Small | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

---

## 📊 参数初始化

### 初始化策略

GPT-2 使用特定的初始化策略来确保训练稳定性：

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            # 对于 output projections，缩小初始化
            std *= (2 * n_layer) ** -0.5
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
```

**关键点**：
- 线性层权重：正态分布，std=0.02
- 带 `NANOGPT_SCALE_INIT` 标记的层：std *= (2 * n_layer)^-0.5
- 偏置：零初始化

**为什么需要缩放？**
- 深层网络中，残差累加可能导致方差放大
- 缩放有助于保持方差稳定

---

## 🔧 训练配置详解

### Batch Size

```
total_batch_size = 524288 (2^19) ≈ 0.5M tokens

在 8 GPU 上：
- 每 GPU micro batch = 64 samples
- sequence length = 1024
- grad_accum = 524288 / (64 * 1024 * 8) = 1
```

### 学习率调度

```
学习率曲线：
      max_lr
        /\
       /  \
      /    \________
     /              \
    /                \
   /                  \___
warmup      cosine decay
```

```python
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

### 优化器配置

```python
optimizer = model.configure_optimizers(
    weight_decay=0.1,      # 权重衰减（仅应用于 2D+ 参数）
    learning_rate=6e-4,    # 最大学习率
    device_type="cuda"     # 用于决定是否使用 fused AdamW
)
```

**AdamW 参数**：
- betas = (0.9, 0.95)
- eps = 1e-8
- fused = True (如果可用)

**权重衰减规则**：
- dim >= 2 的参数：weight_decay = 0.1 (如 Linear weights)
- dim < 2 的参数：weight_decay = 0.0 (如 biases, LayerNorm, Embeddings)

---

## 📈 训练监控

### 关键指标

| 指标 | 含义 | 正常范围 | 异常信号 |
|------|------|----------|----------|
| train loss | 当前 batch 损失 | 下降中 | NaN |
| val loss | 验证集损失 | 稳定下降 | 上升（过拟合） |
| grad norm | 梯度 L2 范数 | 0.5-1.5 | >5 或接近 0 |
| lr | 学习率 | 按调度变化 | - |
| tokens/sec | 吞吐量 | 稳定 | 下降（GPU 问题） |

### 日志输出示例

```
step     0 | loss: 10.791300 | lr 8.4286e-07 | norm: 0.8547 | dt: 1200.00ms | tok/sec: 43690.67
step   100 | loss: 9.812345 | lr 8.5714e-05 | norm: 1.0234 | dt: 150.00ms | tok/sec: 349525.33
step   500 | loss: 4.567890 | lr 4.2857e-04 | norm: 0.8923 | dt: 145.00ms | tok/sec: 361379.54
validation loss: 3.4567
HellaSwag accuracy: 2500/10042=0.2489
```

### 梯度裁剪

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**为什么需要梯度裁剪？**
- 防止梯度爆炸
- 保持训练稳定
- 通常裁剪到 max_norm = 1.0

---

## 🧪 HellaSwag 评估

### 任务定义

给定上下文，选择最合理的结尾。

```
Context: "A man is sitting on a roof. he"
Endings:
  0: "is using wrap to wrap a pair of skis."
  1: "is ripping level tiles off."
  2: "is holding a rubik's cube."
  3: "starts pulling up roofing on a roof."
Label: 3
```

### 评估指标

- **acc**: 基于原始概率的准确率
- **acc_norm**: 基于长度归一化概率的准确率（更常用）

### 评估代码

```python
def get_most_likely_row(tokens, mask, logits):
    # 计算移位损失
    shift_logits = logits[..., :-1, :]
    shift_tokens = tokens[..., 1:]
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction='none'
    )

    # 只看候选结尾部分（mask == 1）
    shift_mask = (mask[..., 1:]).contiguous()
    masked_losses = shift_losses * shift_mask
    avg_loss = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)

    return avg_loss.argmin().item()
```

---

## 📦 数据处理

### FineWeb 数据集

```bash
# 下载和预处理
python fineweb.py

# 生成：
# edu_fineweb10B/
#   ├── edufineweb_val_000000     # 验证集 (100M tokens)
#   ├── edufineweb_train_000001  # 训练集分片 1
#   ├── edufineweb_train_000002  # 训练集分片 2
#   └── ...
```

### 数据加载器

```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        # 分片轮转加载
        # 支持分布式训练
        pass

    def next_batch(self):
        # 返回 x: (B, T), y: (B, T)
        # y[i] = x[i] 向前移动一位（预测下一个 token）
        pass
```

---

## 🚀 分布式训练

### DDP 原理

```
GPU 0: model副本0 ← batch0 → 计算梯度 ←←←→→ 同步梯度 → 更新参数
GPU 1: model副本1 ← batch1 → 计算梯度 ←←←→→ 同步梯度 → 更新参数
...
GPU 7: model副本7 ← batch7 → 计算梯度 ←←←→→ 同步梯度 → 更新参数
```

### 启动命令

```bash
# 8 GPU 训练
torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 4 GPU 训练
torchrun --standalone --nproc_per_node=4 train_gpt2.py
```

### 关键 DDP 代码

```python
# 初始化
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])

# 包装模型
model = DDP(model, device_ids=[ddp_local_rank])

# 梯度同步控制
if ddp:
    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

# 损失聚合
dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
```

---

## 🛠️ 常见问题

### 1. 显存不足 (OOM)

解决方案：
- 减小 B 和 T
- 增大 grad_accum_steps
- 使用 gradient checkpointing
- 减小模型大小

### 2. 损失不下降

检查：
- 学习率是否合适
- 数据是否正确加载
- 模型是否正确初始化
- 是否有 NaN/Inf

### 3. 梯度爆炸

解决：
- 启用梯度裁剪 (clip_grad_norm_)
- 减小学习率
- 检查初始化

### 4. 训练速度慢

优化：
- 使用 bf16 混合精度
- 启用 CUDA graphs
- 检查数据加载是否成为瓶颈
- 使用更快的存储

---

## 📚 扩展阅读

- [GPT-2 论文: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-2 模型卡](https://github.com/openai/gpt-2/blob/master/model_card.md)
- [FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HellaSwag 论文](https://arxiv.org/abs/1905.07830)

---

> 📚 视频: [复现 GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
> 📦 代码: [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)