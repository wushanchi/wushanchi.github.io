"""
Makemore — 字符级语言模型

Andrej Karpathy 的字符级语言模型教程代码。
本文件包含多种语言模型实现：Bigram、MLP、RNN、GRU、BoW、Transformer

本注释版重点讲解 makemore Part 1 的核心内容：
- Bigram 模型（最简单 baseline）
- MLP 语言模型
- CharDataset 数据处理
- 训练循环与采样
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
# 第一部分：配置类
# ============================================================================

@dataclass
class ModelConfig:
    """
    模型配置数据类

    Attributes:
        block_size: 输入序列的最大长度（也是位置编码的大小）
        vocab_size: 词汇表大小（字符数量）
        n_layer: Transformer 层数（仅 Transformer 使用）
        n_embd: 嵌入维度（token 嵌入和位置嵌入的维度）
        n_embd2: 中间层维度（MLP/BoW/RNN 使用）
        n_head: 注意力头数（仅 Transformer 使用）
    """
    block_size: int = None  # 输入序列最大长度
    vocab_size: int = None  # 词汇表大小
    n_layer: int = 4        # Transformer 层数
    n_embd: int = 64        # 嵌入维度
    n_embd2: int = 64       # 隐藏层维度
    n_head: int = 4         # 注意力头数


# ============================================================================
# 第二部分：Bigram 语言模型 — 最简单的 baseline
# ============================================================================

class Bigram(nn.Module):
    """
    Bigram 语言模型 — 史上最简单的语言模型

    原理：只根据前一个字符预测下一个字符
    没有任何嵌入层，只是一个查找表（lookup table）

    维度变化:
        输入: idx (B, T) — 每个元素是 0~vocab_size-1 的整数
        输出: logits (B, T, vocab_size) — 每个位置的 logits 用于预测下一个字符

    参数:
        logits: (vocab_size, vocab_size) 的权重矩阵
        logits[i][j] = 给定字符 i，下一个字符是 j 的 logit

    Example:
        vocab_size = 26（a-z）
        logits[0][1] = 给定 'a'，预测 'b' 的得分
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        # Bigram 模型就是一个 vocab_size × vocab_size 的查找表
        # logits[i] 存储了给定字符 i 时，下一个字符的未归一化分数
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        """
        返回模型能够处理的最大上下文长度
        Bigram 只需要 1 个前驱字符
        """
        return 1

    def forward(self, idx, targets=None):
        """
        前向传播

        Args:
            idx: (B, T) — 字符索引序列
            targets: (B, T) — 目标字符索引（用于计算损失）

        Returns:
            logits: (B, T, vocab_size) — 每个位置的预测 logits
            loss: 交叉熵损失（如果提供了 targets）
        """
        # self.logits[idx] 是索引选择操作
        # 如果 idx[i,j] = k，则选择 logits[k]
        # 结果形状: (B, T, vocab_size)
        logits = self.logits[idx]

        # 计算交叉熵损失
        loss = None
        if targets is not None:
            # logits.reshape(-1, vocab_size): (B*T, vocab_size)
            # targets.reshape(-1): (B*T,)
            # 计算预测下一个字符的损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# ============================================================================
# 第三部分：MLP 语言模型 — Bengio 2003
# ============================================================================

class MLP(nn.Module):
    """
    MLP 语言模型 — 多层感知机语言模型

    原理：
    1. 将前 block_size 个字符的嵌入拼接起来
    2. 通过 MLP 计算下一个字符的 logits

    维度变化:
        输入: idx (B, T) — 字符索引序列
        嵌入: wte(idx) → (B, T, n_embd)
        拼接: cat(embs) → (B, T, block_size * n_embd)
        MLP:  → (B, T, vocab_size)

    Reference:
        Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # Token 嵌入表: 每个字符有一个 n_embd 维的向量
        # vocab_size + 1 是为了特殊 <BLANK> 令牌（在序列开始前插入）
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # MLP: 拼接的嵌入 → 隐藏层 → 输出
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),  # block_size * n_embd → n_embd2
            nn.Tanh(),                                                    # 激活函数
            nn.Linear(config.n_embd2, self.vocab_size)                    # n_embd2 → vocab_size
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        """
        前向传播

        实现细节：
        - 对 block_size 个历史字符分别查嵌入表
        - 每次查完后将 idx 向左滚动 1 位，用 <BLANK> 填充开头
        - 最后拼接所有嵌入
        """
        # 收集所有历史位置的嵌入
        embs = []
        for k in range(self.block_size):
            # 获取当前时间步的嵌入
            tok_emb = self.wte(idx)  # (B, T, n_embd)
            # 将索引向左滚动，为下一个时间步做准备
            idx = torch.roll(idx, 1, 1)
            # 第一个位置填充特殊的 <BLANK> 令牌
            idx[:, 0] = self.vocab_size  # vocab_size 作为特殊令牌索引
            embs.append(tok_emb)

        # 拼接所有嵌入: (B, T, n_embd * block_size)
        x = torch.cat(embs, -1)

        # MLP 前向传播
        logits = self.mlp(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# ============================================================================
# 第四部分：CharDataset — 数据处理
# ============================================================================

class CharDataset(Dataset):
    """
    字符级数据集

    功能：
    1. 将字符序列编码为整数索引
    2. 创建训练样本（输入和目标偏移一位）
    3. 处理词汇表映射（stoi, itos）

    Example:
        词汇表: {'a': 1, 'b': 2, 'c': 3, ...}
        特殊令牌: 0 = <START>, <STOP>

        单词 "abc":
        - 编码: [1, 2, 3]
        - 输入 x: [0, 1, 2, 3]  (<START>, a, b, c)
        - 目标 y: [1, 2, 3, 0]  (a, b, c, <STOP>)
    """

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length

        # 字符到索引的映射（从 1 开始，0 保留给特殊令牌）
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        # 索引到字符的反向映射
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.words)

    def contains(self, word):
        """检查单词是否在训练集中"""
        return word in self.words

    def get_vocab_size(self):
        """
        返回词汇表大小
        包括所有字符 + 1 个特殊令牌（索引 0）
        """
        return len(self.chars) + 1

    def get_output_length(self):
        """
        返回输出序列长度
        包括 <START> 令牌 + 单词长度
        """
        return self.max_word_length + 1

    def encode(self, word):
        """
        将单词编码为索引序列

        Args:
            word: 字符串单词

        Returns:
            torch.tensor: 索引序列
        """
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        """
        将索引序列解码为单词

        Args:
            ix: 索引序列

        Returns:
            str: 解码后的单词
        """
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        """
        获取单个训练样本

        Args:
            idx: 样本索引

        Returns:
            x: (max_word_length + 1,) 输入序列
            y: (max_word_length + 1,) 目标序列（向右偏移一位）
        """
        word = self.words[idx]
        ix = self.encode(word)  # 编码单词

        # 创建输入序列: [0, char1, char2, ..., charN]
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        # 填充输入序列
        x[1:1 + len(ix)] = ix
        # 填充目标序列（向右偏移）
        y[:len(ix)] = ix
        # 目标序列最后一位标记为 -1（将在损失计算中忽略）
        y[len(ix) + 1:] = -1

        return x, y


def create_datasets(input_file):
    """
    创建训练和测试数据集

    Args:
        input_file: 包含单词列表的文件（每行一个单词）

    Returns:
        train_dataset: 训练集
        test_dataset: 测试集（约 10% 的数据）
    """
    # 读取数据
    with open(input_file, 'r') as f:
        data = f.read()

    # 分割为单词列表
    words = data.splitlines()
    words = [w.strip() for w in words]  # 去除首尾空白
    words = [w for w in words if w]      # 去除空字符串

    # 构建字符集
    chars = sorted(list(set(''.join(words))))  # 所有出现的唯一字符
    max_word_length = max(len(w) for w in words)

    print(f"数据集中的样本数: {len(words)}")
    print(f"最大单词长度: {max_word_length}")
    print(f"词汇表中的唯一字符数: {len(chars)}")
    print("词汇表:")
    print(''.join(chars))

    # 划分训练集和测试集
    test_set_size = min(1000, int(len(words) * 0.1))  # 10% 或最多 1000 个样本
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]

    print(f"将数据集分为 {len(train_words)} 个训练样本和 {len(test_words)} 个测试样本")

    # 创建数据集对象
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


# ============================================================================
# 第五部分：采样与评估
# ============================================================================

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    从模型生成下一个字符

    Args:
        model: 语言模型
        idx: (B, T) 条件序列
        max_new_tokens: 要生成的新令牌数量
        temperature: 温度参数（越高越随机，越低越确定）
        do_sample: 是否采样（False 则选择最可能的）
        top_k: 只考虑前 k 个最高概率的令牌

    Returns:
        idx: (B, T + max_new_tokens) 扩展后的序列
    """
    block_size = model.get_block_size()

    for _ in range(max_new_tokens):
        # 如果序列太长，截断到 block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # 前向传播
        logits, _ = model(idx_cond)
        # 只取最后一个时间步的 logits
        logits = logits[:, -1, :] / temperature

        # 可选：只保留 top_k
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 归一化为概率
        probs = F.softmax(logits, dim=-1)

        # 采样或选择最可能的
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(model, dataset, train_dataset, test_dataset, num=10):
    """
    从模型采样并打印生成的单词
    """
    # 初始化输入（全为 0，即 <START> 令牌）
    X_init = torch.zeros(num, 1, dtype=torch.long)
    top_k = args.top_k if args.top_k != -1 else None

    # 计算需要生成的长度
    steps = dataset.get_output_length() - 1  # 减去 <START> 令牌

    # 生成样本
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    # 分类样本
    train_samples, test_samples, new_samples = [], [], []

    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()  # 去掉 <START> 令牌

        # 在 <STOP> 令牌处截断
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]

        word_samp = dataset.decode(row)

        # 分类
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)

    # 打印结果
    print('-' * 80)
    for lst, desc in [(train_samples, '训练集'), (test_samples, '测试集'), (new_samples, '新生成')]:
        print(f"{len(lst)} 个样本在 {desc}:")
        for word in lst:
            print(word)
    print('-' * 80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    """
    评估模型在数据集上的损失
    """
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


# ============================================================================
# 第六部分：InfiniteDataLoader — 无限数据加载器
# ============================================================================

class InfiniteDataLoader:
    """
    无限数据加载器

    通过替换采样器实现无限迭代
    用于训练时持续产生批次而不耗尽数据
    """

    def __init__(self, dataset, **kwargs):
        # 替换采样器：有放回地无限采样
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        """获取下一个批次"""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


# ============================================================================
# 第七部分：主训练循环
# ============================================================================

if __name__ == '__main__':

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Make More")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt',
                        help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out',
                        help="output working directory")
    parser.add_argument('--resume', action='store_true',
                        help="resume optimization from existing model")
    parser.add_argument('--sample-only', action='store_true',
                        help="just sample from the model and quit")
    parser.add_argument('--num-workers', '-n', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--max-steps', type=int, default=-1,
                        help="max number of optimization steps")
    parser.add_argument('--device', type=str, default='cpu',
                        help="device to use for compute")
    parser.add_argument('--seed', type=int, default=3407,
                        help="seed")
    parser.add_argument('--top-k', type=int, default=-1,
                        help="top-k for sampling, -1 means no top-k")
    parser.add_argument('--type', type=str, default='transformer',
                        help="model type: bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4,
                        help="number of layers")
    parser.add_argument('--n-head', type=int, default=4,
                        help="number of heads")
    parser.add_argument('--n-embd', type=int, default=64,
                        help="number of feature channels")
    parser.add_argument('--n-embd2', type=int, default=64,
                        help="number of feature channels elsewhere")
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help="batch size")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01,
                        help="weight decay")

    args = parser.parse_args()
    print(vars(args))

    # 系统初始化
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # 创建数据集
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"数据集: vocab_size={vocab_size}, block_size={block_size}")

    # 创建模型
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2
    )

    if args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    # ... 其他模型类型

    model.to(args.device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 加载已有模型（如果 resume）
    if args.resume or args.sample_only:
        print("从现有模型恢复")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))

    if args.sample_only:
        print_samples(model, train_dataset, train_dataset, test_dataset, num=50)
        sys.exit()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    # 无限数据加载器
    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # 训练循环
    best_loss = None
    step = 0

    while True:
        t0 = time.time()

        # 获取批次
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # 前向传播
        logits, loss = model(X, Y)

        # 反向传播
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # 日志
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1 - t0) * 1000:.2f}ms")

        # 评估
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")

            # 保存最佳模型
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} 是目前最佳，保存模型到 {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # 采样
        if step > 0 and step % 200 == 0:
            print_samples(model, train_dataset, train_dataset, test_dataset, num=10)

        step += 1

        # 终止条件
        if args.max_steps >= 0 and step >= args.max_steps:
            break