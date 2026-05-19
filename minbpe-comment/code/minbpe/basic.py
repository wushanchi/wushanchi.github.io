"""
BasicTokenizer - 最简单的 BPE (Byte Pair Encoding) Tokenizer

本模块实现了一个基础的 BPE 算法的 tokenizer。

算法参考了 GPT tokenizer 的实现:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

与 RegexTokenizer 的区别:
- 不处理正则表达式分割模式
- 不处理特殊 token
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """
    最简单版本的 BPE Tokenizer。

    工作原理:
    1. 将输入文本转换为 UTF-8 字节序列（0-255 的整数）
    2. 迭代查找最频繁出现的相邻字节对
    3. 将该字节对合并为一个新的 token（分配一个新的 ID）
    4. 重复步骤 2-3，直到达到目标词汇表大小

    特点:
    - 无正则表达式预处理
    - 无特殊 token 支持
    - 最简单最轻量
    """

    def __init__(self):
        """初始化 BasicTokenizer，调用父类构造函数。"""
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        训练 tokenizer，从文本构建大小为 vocab_size 的词汇表。

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小（必须 >= 256）
            verbose: 是否打印详细训练过程信息

        算法:
        1. 将文本编码为 UTF-8 字节序列
        2. 迭代执行合并操作 num_merges 次
        3. 每次找到最频繁的字节对，合并为新 token

        合并规则:
        - 初始 vocab 包含 256 个单字节 (0-255)
        - 每次合并添加一个新 token（从 256 开始编号）
        - vocab_size = 256 + num_merges
        """
        assert vocab_size >= 256, "词汇表大小必须至少为 256（所有单字节）"
        num_merges = vocab_size - 256  # 需要执行的合并次数

        # 步骤 1: 将文本转换为 UTF-8 字节序列
        text_bytes = text.encode("utf-8")  # 原始字节
        ids = list(text_bytes)  # 转换为 0-255 整数列表

        # 步骤 2: 迭代合并最常见的字节对
        merges = {}  # (int, int) -> int, 合并规则字典
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes, vocab 初始化

        for i in range(num_merges):
            # 统计所有相邻字节对的出现频率
            stats = get_stats(ids)

            # 找到出现频率最高的字节对
            pair = max(stats, key=stats.get)

            # 为新 token 分配下一个可用 ID（从 256 开始）
            idx = 256 + i

            # 将所有出现的该字节对替换为新 token ID
            ids = merge(ids, pair, idx)

            # 保存合并规则
            merges[pair] = idx

            # 更新 vocab：新 token = 两个被合并 token 的字节拼接
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 打印详细信息（如果 verbose=True）
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存为类变量
        self.merges = merges  # 用于 encode()
        self.vocab = vocab    # 用于 decode()

    def decode(self, ids):
        """
        将 token ID 列表解码回字符串。

        参数:
            ids: token ID 列表（整数列表）

        返回:
            解码后的 Python 字符串

        算法:
        1. 通过 vocab 将每个 token ID 转换回字节序列
        2. 拼接所有字节序列
        3. 用 UTF-8 解码为字符串
        """
        # 使用 vocab 将每个 ID 转换为字节，然后拼接
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # 解码为字符串，错误字符用替换符
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        将字符串编码为 token ID 列表。

        参数:
            text: 输入字符串

        返回:
            token ID 列表（整数列表）

        算法:
        1. 将文本转换为 UTF-8 字节序列
        2. 迭代应用所有合并规则：
           - 找到当前序列中索引最小的可合并字节对
           - 用对应的新 token ID 替换该字节对
        3. 返回最终的 token ID 列表

        注意:
        - 使用贪婪算法，每次选择合并索引最小的对
        - 这确保了编码的确定性
        """
        # 将文本转换为 UTF-8 字节的整数表示
        text_bytes = text.encode("utf-8")  # 原始字节
        ids = list(text_bytes)  # 转换为 0-255 整数列表

        # 迭代合并，只要序列长度 >= 2 就继续
        while len(ids) >= 2:
            # 统计当前序列中所有字节对的频率
            stats = get_stats(ids)

            # 找到合并索引最小的字节对
            # （索引越小表示越早被合并，所以在序列中位置越靠前）
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # 如果字节对不在 merges 中，说明没有更多可合并的对
            # 这种情况下，min 会返回 inf 作为值
            if pair not in self.merges:
                break  # 无法再合并，退出循环

            # 执行合并：用新 token ID 替换该字节对
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids