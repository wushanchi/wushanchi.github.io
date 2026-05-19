"""
hellaswag.py - HellaSwag 数据集下载与评估脚本
============================================

HellaSwag 是一个用于评估语言模型 commonsense reasoning 能力的基准数据集。
https://github.com/rowanz/hellaswag

数据集结构：
- 每个样本提供一个上下文（context）和 4 个候选结尾（endings）
- 其中只有 1 个结尾是正确的
- 模型需要选出最合理的结尾

评估方式：
- 计算每个候选结尾的条件概率/损失
- 选择损失最低的结尾作为预测结果

支持的指标：
- acc：标准准确率（基于原始损失）
- acc_norm：标准化准确率（基于归一化后的概率）

数据集统计：
- 验证集：10,042 个样本
- 训练集：39,905 个样本
- 测试集：10,042 个样本

GPT-2 性能参考：
- gpt2 (124M): acc ≈ 28.92%, acc_norm ≈ 31.14%
- gpt2-xl (1558M): acc ≈ 40.04%, acc_norm ≈ 50.89%

使用方法：
    # 评估预训练模型
    python hellaswag.py -m gpt2 -d cuda

    # 或在训练脚本中调用
    from hellaswag import iterate_examples, render_example
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# =============================================================================
# 常量
# =============================================================================

# 数据缓存目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

# HellaSwag 数据集 URL
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# 初始化 GPT-2 tokenizer（用于编码文本）
enc = tiktoken.get_encoding("gpt2")

# =============================================================================
# 数据下载
# =============================================================================

def download_file(url: str, fname: str, chunk_size=1024):
    """
    从 URL 下载文件并保存到本地

    参数:
        url: 下载链接
        fname: 保存路径
        chunk_size: 每次读取的块大小（字节）
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split):
    """
    下载指定分割的 HellaSwag 数据集

    参数:
        split: 'train', 'val', 或 'test'
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")

    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


# =============================================================================
# 数据渲染（将 JSON 转换为 tensors）
# =============================================================================

def render_example(example):
    """
    将 HellaSwag 样本渲染为 PyTorch 张量

    每个样本包含：
    - ctx: 上下文（一个句子或段落）
    - endings: 4 个候选结尾
    - label: 正确答案的索引 (0, 1, 2, 或 3)

    返回：
    - data: 包含元数据的字典
    - tokens: shape = (4, max_len)，4 个候选的 token 序列（padding 到相同长度）
    - mask: shape = (4, max_len)，1 表示该位置属于候选结尾（用于计算损失）
    - label: 正确答案的索引

    重要：mask 的作用是只计算候选结尾部分的损失，
    不考虑上下文部分，因为不同候选结尾的上下文是相同的。
    """
    ctx = example["ctx"]          # 上下文
    label = example["label"]       # 正确答案索引
    endings = example["endings"]   # 4 个候选结尾

    # 用于重现此评估的数据
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # 编码上下文
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    # 编码每个候选结尾，并收集 tokens 和 mask
    tok_rows = []      # 4 个候选的完整 token 序列
    mask_rows = []     # 4 个候选的 mask（1 表示候选结尾部分）

    for end in endings:
        # 注意：GPT-2 tokenizer 需要在结尾前添加空格
        end_tokens = enc.encode(" " + end)

        # 完整序列 = 上下文 tokens + 结尾 tokens
        tok_rows.append(ctx_tokens + end_tokens)

        # mask: 上下文部分为 0，结尾部分为 1
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        data["ending_tokens"].append(end_tokens)

    # 处理不同候选结尾长度不一致的问题（padding 到相同长度）
    max_len = max(len(row) for row in tok_rows)

    # 创建 padded tensors
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    """
    迭代指定分割的所有 HellaSwag 样本

    参数:
        split: 'train', 'val', 或 'test'

    生成器：
        yield example: 每个样本（字典格式）
    """
    # 确保数据已下载
    download(split)

    # 逐行读取 JSONL 文件
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


# =============================================================================
# 评估函数
# =============================================================================

@torch.no_grad()
def evaluate(model_type, device):
    """
    评估预训练模型在 HellaSwag 上的性能

    参数:
        model_type: Hugging Face 模型名称（如 'gpt2', 'gpt2-medium' 等）
        device: 设备（如 'cuda', 'cpu'）
    """
    # 设置矩阵乘法精度（使用 tf32 可加速）
    torch.set_float32_matmul_precision('high')

    # 加载预训练模型
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    # 可选：使用 torch.compile 加速
    # model = torch.compile(model)

    num_correct_norm = 0  # 标准化正确数
    num_correct = 0       # 标准正确数
    num_total = 0          # 总数

    for example in iterate_examples("val"):
        # 渲染样本为 tensors
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # 获取模型 logits
        logits = model(tokens).logits

        # 计算移位损失（预测下一个 token）
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()

        # 展平
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        # 计算每个位置的损失
        shift_losses = F.cross_entropy(
            flat_shift_logits,
            flat_shift_tokens,
            reduction='none'
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # 只对候选结尾部分计算平均损失
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # 求和并平均
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # 找出损失最低的候选
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # 累积统计
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # 调试：打印前几个样本的详细信息
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")


# =============================================================================
# 命令行接口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HellaSwag 评估脚本")
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        default="gpt2",
        help="模型类型（如 gpt2, gpt2-medium, gpt2-large, gpt2-xl）"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda",
        help="设备（如 cuda, cpu）"
    )
    args = parser.parse_args()

    evaluate(args.model_type, args.device)


# =============================================================================
# 数据格式说明
# =============================================================================
#
# HellaSwag JSON item 示例：
# {
#     "ind": 24,
#     "activity_label": "Roof shingle removal",
#     "ctx_a": "A man is sitting on a roof.",
#     "ctx_b": "he",
#     "ctx": "A man is sitting on a roof. he",
#     "split": "val",
#     "split_type": "indomain",
#     "label": 3,
#     "endings": [
#         "is using wrap to wrap a pair of skis.",
#         "is ripping level tiles off.",
#         "is holding a rubik's cube.",
#         "starts pulling up roofing on a roof."
#     ],
#     "source_id": "activitynet~v_-JhWjGDPHMY"
# }
#
# 字段说明：
# - ind: 数据集 ID
# - activity_label: ActivityNet 或 WikiHow 标签
# - ctx_a: 上下文（上半部分，可能是完整的句子）
# - ctx_b: 上下文（下半部分，不完整的短语）
# - ctx: 完整的上下文 = ctx_a + " " + ctx_b
# - endings: 4 个候选结尾
# - label: 正确答案的索引 (0-3)
# - split: train, val, 或 test
# - split_type: indomain（训练时见过类似活动）或 zeroshot（未见过的活动）
# - source_id: 来源视频或 WikiHow 文章 ID
#