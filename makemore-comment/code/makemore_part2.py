"""
Makemore Part 2 — MLP 语言模型

本代码实现多层感知机（MLP）字符级语言模型，来自 Bengio 2003 论文。

来源: https://github.com/karpathy/makemore
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ModelConfig:
    """
    模型配置

    Attributes:
        block_size: 输入序列最大长度
        vocab_size: 词汇表大小
        n_embd: 嵌入维度
        n_embd2: MLP 隐藏层维度
    """
    block_size: int = None  # 输入序列最大长度
    vocab_size: int = None # 词汇表大小
    n_layer: int = 4       # Transformer 层数（用于 Transformer 模型）
    n_head: int = 4        # 注意力头数
    n_embd: int = 64        # 嵌入维度
    n_embd2: int = 64       # MLP 隐藏层维度


# ============================================================================
# MLP 语言模型 — Bengio 2003
# ============================================================================

class MLP(nn.Module):
    """
    MLP 语言模型

    原理：将前 block_size 个字符的嵌入拼接，通过 MLP 预测下一个字符

    架构：
    ┌─────────────────────────────────────────────────────────────┐
    │  输入: idx (B, T) — 字符索引序列                            │
    │      ↓                                                     │
    │  Token 嵌入层: wte(idx) → (B, T, n_embd)                   │
    │      ↓                                                     │
    │  滚动拼接: 收集 block_size 个历史嵌入                       │
    │      ↓                                                     │
    │  Flatten: (B, T, n_embd * block_size)                     │
    │      ↓                                                     │
    │  Linear: n_embd*block_size → n_embd2                      │
    │      ↓                                                     │
    │  Tanh: 激活函数                                            │
    │      ↓                                                     │
    │  Linear: n_embd2 → vocab_size                             │
    │      ↓                                                     │
    │  输出: logits (B, T, vocab_size)                           │
    └─────────────────────────────────────────────────────────────┘

    Reference: Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # Token 嵌入表
        # vocab_size + 1: 加 1 给特殊的 <BLANK> 令牌（在序列开始前插入）
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # MLP: 拼接的嵌入 → 隐藏层 → 输出
        self.mlp = nn.Sequential(
            # 第一层: n_embd * block_size → n_embd2
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            # 激活函数
            nn.Tanh(),
            # 第二层: n_embd2 → vocab_size
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        """返回模型能够处理的最大上下文长度"""
        return self.block_size

    def forward(self, idx, targets=None):
        """
        前向传播

        Args:
            idx: (B, T) — 字符索引序列
            targets: (B, T) — 目标序列（用于计算损失）

        Returns:
            logits: (B, T, vocab_size) — 每个位置的预测 logits
            loss: 交叉熵损失（如果提供了 targets）
        """
        # 收集所有历史位置的嵌入
        embs = []
        for k in range(self.block_size):
            # 获取当前时间步的嵌入
            tok_emb = self.wte(idx)  # (B, T, n_embd)

            # 将索引向左滚动 1 位，为下一个时间步做准备
            # 例如: [a, b, c] → [_, a, b] (_ 是空白位置)
            idx = torch.roll(idx, 1, 1)

            # 第一个位置填充特殊的 <BLANK> 令牌
            # vocab_size 索引被用作特殊令牌
            idx[:, 0] = self.vocab_size

            embs.append(tok_emb)

        # 拼接所有嵌入: (B, T, n_embd * block_size)
        x = torch.cat(embs, -1)

        # MLP 前向传播
        logits = self.mlp(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss


# ============================================================================
# CharDataset — 字符级数据集
# ============================================================================

class CharDataset(Dataset):
    """
    字符级数据集

    功能：
    1. 将字符序列编码为整数索引
    2. 创建训练样本（输入和目标偏移一位）

    Example:
        单词 "abc":
        - 编码: [1, 2, 3]
        - 输入 x: [0, 1, 2, 3]  (<START>, a, b, c)
        - 目标 y: [1, 2, 3, 0]  (a, b, c, <STOP>)
    """

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length

        # 字符到索引映射（从 1 开始，0 保留给特殊令牌）
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        # 反向映射
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)

        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1  # 忽略填充位置

        return x, y


def create_datasets(input_file):
    """创建训练和测试数据集"""
    with open(input_file, 'r') as f:
        data = f.read()

    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]

    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)

    print(f"数据集样本数: {len(words)}")
    print(f"最大单词长度: {max_word_length}")
    print(f"词汇表大小: {len(chars)}")
    print(f"词汇表: {''.join(chars)}")

    # 划分训练集和测试集
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]

    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


# ============================================================================
# 辅助函数
# ============================================================================

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """自回归生成下一个字符"""
    block_size = model.get_block_size()

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(model, dataset, train_dataset, test_dataset, num=10, top_k=None):
    """采样并打印生成的单词"""
    X_init = torch.zeros(num, 1, dtype=torch.long)
    steps = dataset.get_output_length() - 1

    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, test_samples, new_samples = [], [], []

    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = dataset.decode(row)

        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)

    print('-' * 80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-' * 80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    """评估模型在数据集上的损失"""
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)

    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())

        if max_batches is not None and i >= max_batches:
            break

    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss


class InfiniteDataLoader:
    """无限数据加载器"""
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


# ============================================================================
# 主训练循环
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make More - MLP")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir', '-o', type=str, default='out')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--n-embd', type=int, default=64)
    parser.add_argument('--n-embd2', type=int, default=64)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4)
    parser.add_argument('--max-steps', type=int, default=-1)

    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # 创建数据集
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    # 创建模型
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2
    )
    model = MLP(config)
    model.to(args.device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 训练循环
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)

    best_loss = None
    step = 0

    while True:
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | time {(t1 - t0) * 1000:.2f}ms")

        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")

            if best_loss is None or test_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(args.work_dir, 'model.pt'))
                best_loss = test_loss

        if step > 0 and step % 200 == 0:
            print_samples(model, train_dataset, train_dataset, test_dataset, num=10, top_k=args.top_k if args.top_k != -1 else None)

        step += 1
        if args.max_steps >= 0 and step >= args.max_steps:
            break