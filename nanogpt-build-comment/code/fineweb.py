"""
fineweb.py - FineWeb-Edu 数据集下载与预处理脚本
===============================================

本脚本用于：
1. 从 Hugging Face 下载 FineWeb-Edu 数据集
2. 使用 GPT-2 tokenizer 对文本进行分词
3. 将分词后的数据保存为二进制分片文件

FineWeb-Edu 是一个高质量的教育文本数据集，包含约 100 亿个 token。
数据被分割成多个分片，每个分片包含约 1 亿个 token。

使用方法：
    python fineweb.py

输出：
    edu_fineweb10B/ 目录下的多个 .npy 文件
    - edufineweb_val_000000 (第一个分片作为验证集)
    - edufineweb_train_000001, 000002, ... (训练集分片)
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm             # pip install tqdm

# =============================================================================
# 配置
# =============================================================================

# 本地缓存目录（相对于当前脚本目录）
local_dir = "edu_fineweb10B"

# Hugging Face 数据集名称
# sample-10BT 表示采样 100 亿个 token
remote_name = "sample-10BT"

# 每个分片的大小（以 token 数量计）
# 100M tokens per shard，总共约 100 个分片
shard_size = int(1e8)

# 创建本地缓存目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# =============================================================================
# 下载数据集
# =============================================================================

# 加载 FineWeb-Edu 数据集
# 这会从 Hugging Face 下载数据集到本地缓存
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# =============================================================================
# 初始化 tokenizer
# =============================================================================

# 获取 GPT-2 的 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 获取特殊token：<|endoftext|> (用于分隔文档)
eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    """
    对单个文档进行分词

    该函数将一个文档（一个样本）转换为 token 序列。

    处理流程：
    1. 在文档开头添加 <|endoftext|> token（作为文档分隔符）
    2. 使用 GPT-2 tokenizer 的 encode_ordinary 方法对文本进行编码
       （注意：不处理特殊 token）
    3. 转换为 uint16 类型的 numpy 数组

    参数:
        doc: 字典对象，包含 'text' 字段（文档文本）

    返回:
        tokens_np_uint16: numpy uint16 数组，包含 token IDs
    """
    tokens = [eot]  # 特殊 <|endoftext|> token 分隔所有文档

    # 使用 encode_ordinary 编码文本（不处理特殊 token）
    tokens.extend(enc.encode_ordinary(doc["text"]))

    # 转换为 numpy 数组
    tokens_np = np.array(tokens)

    # 确保 token IDs 在 uint16 范围内（0-65535）
    # GPT-2 的词汇表大小是 50257，所以肯定在范围内
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), \
        "token dictionary too large for uint16"

    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    """
    将 token 序列保存为 .npy 文件

    参数:
        filename: 输出文件路径
        tokens_np: numpy 数组，包含 token IDs
    """
    np.save(filename, tokens_np)


# =============================================================================
# 主处理循环
# =============================================================================

# 获取 CPU 核心数（用于并行处理）
nprocs = max(1, os.cpu_count() // 2)

# 使用多进程池并行处理文档
with mp.Pool(nprocs) as pool:
    shard_index = 0  # 当前分片索引

    # 预分配缓冲区来存储当前分片的数据
    # 使用空数组然后逐步填充会更高效
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0        # 当前分片中的 token 数量
    progress_bar = None    # 进度条

    # 使用 imap 并行处理文档（chunksize=16 表示每次传递给进程 16 个文档）
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # 检查当前分片是否有足够的空间存储新 tokens
        if token_count + len(tokens) < shard_size:
            # 空间足够：直接追加到当前分片
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)

            # 更新进度条
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size,
                    unit="tokens",
                    desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))

        else:
            # 空间不足：写入当前分片，开始新分片

            # 计算剩余空间
            remainder = shard_size - token_count

            # 更新进度条至满
            progress_bar.update(remainder)

            # 将尽可能多的 tokens 填充到当前分片
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

            # 写入当前分片到磁盘
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np)

            # 移动到下一个分片
            shard_index += 1
            progress_bar = None

            # 将剩余的 tokens 填充到新分片
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # 处理最后一个分片（如果还有剩余 tokens）
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(
            DATA_CACHE_DIR,
            f"edufineweb_{split}_{shard_index:06d}"
        )
        write_datafile(filename, all_tokens_np[:token_count])

# =============================================================================
# 输出说明
# =============================================================================
#
# 生成的文件结构：
# edu_fineweb10B/
#     ├── edufineweb_val_000000    # 第一个分片（100M tokens）作为验证集
#     ├── edufineweb_train_000001  # 训练集分片 1
#     ├── edufineweb_train_000002  # 训练集分片 2
#     └── ...
#
# 每个文件：
#     - 格式：.npy (NumPy 二进制格式)
#     - dtype：uint16 (无符号 16 位整数)
#     - 大小：约 200MB (100M tokens × 2 bytes)
#
# 在训练脚本中使用：
#     from fineweb import load_tokens
#     tokens = load_tokens("edu_fineweb10B/edufineweb_train_000001")
#     # tokens shape: (100000000,), dtype: torch.long
#