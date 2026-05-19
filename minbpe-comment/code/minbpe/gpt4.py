"""
GPT4Tokenizer - GPT-4 Tokenizer 的轻量包装实现

本模块实现了 GPT-4 使用的 tokenizer，作为 RegexTokenizer 的包装。

重要说明:
- 这是一个预训练的 tokenizer，在 __init__() 中从 tiktoken 加载预训练的 'cl100k_base'
- 它不是从头训练，而是加载 GPT-4 训练好的词汇表和合并规则
- 由于 GPT-4 的 tokenizer 有一些特殊的历史遗留问题，需要特殊处理

主要特殊处理:
1. 字节顺序洗牌 (byte_shuffle): GPT-4 对单个字节的 token ID 进行了重新排列
2. 需要从 mergeable_ranks 中恢复合并规则

参考:
- https://github.com/openai/tiktoken
- https://github.com/openai/tiktoken/issues/60
- https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
"""

import tiktoken
from .regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    """
    辅助函数，用于在 get_gpt4_merges() 中重建合并森林。

    给定一个 token 和最大排名限制，计算该 token 的合并结构。

    参数:
        mergeable_ranks: 字典，字节序列 -> token ID
        token: 字节序列（bytes）
        max_rank: 最大排名限制，用于控制合并的深度

    返回:
        合并后的字节序列列表

    算法:
    1. 将 token 分解为单字节列表
    2. 迭代查找可合并的相邻字节对
    3. 选择排名最小（优先级最高）的对进行合并
    4. 当没有可合并的对，或排名超过 max_rank 时停止

    这个函数模拟了 BPE 编码过程，但用于反向工程 GPT-4 的合并规则。
    """
    # 将 token 分解为单字节列表
    parts = [bytes([b]) for b in token]

    while True:
        # 查找可合并的对
        min_idx = None
        min_rank = None

        # 遍历所有相邻字节对
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # 计算该对的合并排名
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # 如果没有可合并的对，或者超过最大排名限制，停止
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        assert min_idx is not None

        # 执行合并：将相邻的两个部分合并为一个
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]

    return parts


def recover_merges(mergeable_ranks):
    """
    从 tiktoken 的 mergeable_ranks 中恢复合并规则。

    mergeable_ranks 已经是合并后的字节序列及其 ID。
    我们需要反向工程出原始的合并规则（即哪些对被合并了）。

    参数:
        mergeable_ranks: tiktoken 的合并排名字典

    返回:
        合并规则字典: (idx1, idx2) -> merged_idx

    算法:
    1. 遍历所有 token（按排名顺序）
    2. 对于每个多字节 token，使用 bpe() 函数找出它的合并结构
    3. 恢复原始子 token 的 ID
    4. 构建合并规则

    参考:
    - https://github.com/openai/tiktoken/issues/60
    - https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    """
    merges = {}

    # 按排名顺序遍历所有 token
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # 跳过单字节（叶子节点，不是合并结果）

        # 使用 bpe() 函数获取该 token 的合并结构
        # max_rank=rank 确保我们只考虑优先级高于或等于当前 rank 的合并
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2, f"预期的合并结果为两个部分，实际为 {len(pair)} 个"

        # 恢复组成该合并的两个 token 的 ID
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]

        # 记录合并规则: (子token1_id, 子token2_id) -> 合并token_id
        merges[(ix0, ix1)] = rank

    return merges


# GPT-4 的文本分割模式（与 RegexTokenizer 中的相同）
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-4 使用的特殊 token
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,   # 文本结束标记
    '<|fim_prefix|>': 100258,  # FIM（Fill-in-middle）前缀
    '<|fim_middle|>': 100259,   # FIM 中间部分
    '<|fim_suffix|>': 100260,   # FIM 后缀
    '<|endofprompt|>': 100276   # 提示结束标记
}


class GPT4Tokenizer(RegexTokenizer):
    """
    轻量包装的 GPT-4 Tokenizer，匹配 GPT-4 的 tokenizer 行为。

    工作原理:
    1. 继承 RegexTokenizer 的所有功能
    2. 在 __init__() 中从 tiktoken 加载预训练的词汇表和合并规则
    3. 应用 GPT-4 的特殊处理（字节顺序洗牌）

    主要属性:
    - merges: 合并规则字典
    - vocab: token ID 到字节序列的映射
    - byte_shuffle: 字节顺序洗牌映射（GPT-4 的历史遗留问题）
    - inverse_byte_shuffle: 字节顺序洗牌的逆映射
    """

    def __init__(self):
        """
        初始化 GPT4Tokenizer。

        加载预训练的 'cl100k_base' tokenizer 并应用 GPT-4 的特殊处理。
        """
        super().__init__(pattern=GPT4_SPLIT_PATTERN)

        # 获取官方的 tokenizer 及其合并排名
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks

        # 从 mergeable_ranks 中恢复合并规则
        self.merges = recover_merges(mergeable_ranks)

        # 从合并规则重建 vocab
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        # 这里有另一个棘手的问题：
        # 不知为何，对应单个字节的 token ID 是按不同顺序排列的。
        # 这完全没有逻辑，可能只是历史原因，我们必须处理这个问题。
        # 这就是"字节顺序洗牌"——GPT-4 tokenizer 的一个特殊处理。
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        # 注册 GPT-4 的特殊 token
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        """
        对一个文本块进行编码。

        在处理字节之前，需要先对其进行字节顺序洗牌。

        参数:
            text_bytes: 原始字节序列

        返回:
            token ID 列表

        处理流程:
        1. 应用字节顺序洗牌
        2. 调用父类的 _encode_chunk 进行 BPE 编码
        """
        # 在开始处理字节之前，需要先对其进行洗牌
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        # 调用父类（RegexTokenizer）的编码方法
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        """
        将 token ID 列表解码为字符串。

        参数:
            ids: token ID 列表

        返回:
            解码后的字符串

        处理流程:
        1. 将 token ID 转换为字节序列
        2. 应用字节顺序洗牌的逆操作
        3. 用 UTF-8 解码为字符串
        """
        # 将 ID 转换为字节
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # 应用反向字节顺序洗牌
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        # 解码为字符串
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text, vocab_size, verbose=False):
        """
        训练方法（不支持）。

        GPT4Tokenizer 是一个预训练的 tokenizer，不支持从零开始训练。

        抛出:
            NotImplementedError: 调用此方法时抛出异常
        """
        raise NotImplementedError("GPT4Tokenizer 是预训练的 tokenizer，不支持训练")

    def save(self, file_prefix):
        """
        保存方法（不支持）。

        保存/加载需要一些额外考虑。
        需要修改 base 类的 save/load 以支持 byte_shuffle...
        或者可以将 byte_shuffle 移到基类，但这会让我们的
        美丽的 Tokenizer 为了支持 GPT-4 tokenizer 及其奇怪的字节顺序洗牌历史问题
        而变得丑陋。

        抛出:
            NotImplementedError: 调用此方法时抛出异常
        """
        raise NotImplementedError("GPT4Tokenizer 无法保存")

    def load(self, model_file):
        """
        加载方法（不支持）。

        抛出:
            NotImplementedError: 调用此方法时抛出异常
        """
        raise NotImplementedError("GPT4Tokenizer 无法加载")

    def save_vocab(self, vocab_file):
        """
        保存 vocab 文件（仅用于可视化）。

        以与基类相同的格式输出 GPT-4 的 token，
        供人类查看。

        参数:
            vocab_file: 输出文件路径

        用法:
            python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"

        处理流程:
        1. 应用反向字节顺序洗牌构建 vocab
        2. 合并字节并写入文件
        """
        from .base import render_token

        # 构建 vocab 时考虑字节顺序洗牌
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        # 合并字节并写入文件
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    # 如果是合并结果，以 [sub1][sub2] -> [result] idx 格式输出
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则是叶子 token，直接输出
                    f.write(f"[{s}] {idx}\n")