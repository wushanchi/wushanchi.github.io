"""
RegexTokenizer - 带正则表达式预处理的 BPE Tokenizer

本模块在 BasicTokenizer 的基础上增加了：
- 正则表达式文本分割预处理
- 特殊 token 支持

算法参考了 GPT tokenizer 的实现:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import regex as re
from .base import Tokenizer, get_stats, merge


# -----------------------------------------------------------------------------
# GPT-2/GPT-4 的文本分割模式

# GPT-2 使用的分割模式
# 规则说明:
# - '(?:[sdmt]|ll|ve|re)' 匹配英文缩写（I'm, I'm, it's 等的简化形式）
# - ?\p{L}+ 匹配一个或多个字母
# - ?\p{N}+ 匹配一个或多个数字
# - ?[^\s\p{L}\p{N}]+ 匹配非空白非字母数字的字符（标点等）
# - \s+(?!\S) 匹配空白字符（但不是前向否定，匹配单个空格）
# - \s+ 匹配一个或多个空白字符
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-4 使用的分割模式（略有不同）
# 规则说明:
# - '(?i:[sdmt]|ll|ve|re)' 大小写不敏感的缩写匹配
# - [^\r\n\p{L}\p{N}]?+\p{L}+ 匹配可选的单字符（非字母数字换行）+ 字母序列
# - \p{N}{1,3} 匹配 1-3 个数字
# - ?[^\s\p{L}\p{N}]++[\r\n]* 匹配非空白非字母数字字符（贪婪）+ 可选换行
# - \s*[\r\n] 匹配零个或多个空白 + 换行符
# - \s+(?!\S) 匹配空白（后面不是非空白字符，即单个空格）
# - \s+ 匹配多个空白
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    带正则表达式预处理的 BPE Tokenizer。

    在 BasicTokenizer 的基础上增加了：
    1. 使用正则表达式将文本分割成有意义的块
    2. 支持特殊 token（如 <|endoftext|>）

    工作流程:
    1. 使用正则模式将文本分割成"文本块"
    2. 对每个文本块单独进行 BPE 编码
    3. 处理特殊 token（特殊token不进行BPE编码，直接映射为对应ID）
    4. 拼接所有块的编码结果
    """

    def __init__(self, pattern=None):
        """
        初始化 RegexTokenizer。

        参数:
            pattern: 可选的字符串，覆盖默认的 GPT-4 分割模式
                    如果为 None，则使用 GPT4_SPLIT_PATTERN
        """
        super().__init__()
        # 设置分割模式：默认使用 GPT-4 模式，否则使用提供的模式
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # 编译正则表达式，提高匹配效率
        self.compiled_pattern = re.compile(self.pattern)
        # 特殊 token 字典: token字符串 -> token ID
        self.special_tokens = {}
        # 反转的特殊 token 字典: token ID -> token字符串
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        训练 tokenizer，从文本构建大小为 vocab_size 的词汇表。

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小（必须 >= 256）
            verbose: 是否打印详细训练过程信息

        与 BasicTokenizer 的区别:
        - 首先使用正则模式将文本分割成文本块
        - 对文本块分别进行 BPE 训练
        """
        assert vocab_size >= 256, "词汇表大小必须至少为 256"
        num_merges = vocab_size - 256

        # 步骤 1: 使用正则模式将文本分割成文本块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 步骤 2: 预处理输入文本
        # 将每个文本块转换为 UTF-8 字节序列（整数列表）
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 步骤 3: 迭代合并最常见的字节对
        merges = {}  # (int, int) -> int, 合并规则
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes

        for i in range(num_merges):
            # 统计所有文本块中所有相邻字节对的出现频率
            stats = {}
            for chunk_ids in ids:
                # 传入 stats 字典，get_stats 会就地更新它，累加计数
                get_stats(chunk_ids, stats)

            # 找到出现频率最高的字节对
            pair = max(stats, key=stats.get)

            # 为新 token 分配下一个可用 ID
            idx = 256 + i

            # 将所有文本块中的该字节对替换为新 token ID
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # 保存合并规则
            merges[pair] = idx

            # 更新 vocab
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 打印详细信息（如果 verbose=True）
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存为类变量
        self.merges = merges  # 用于 encode()
        self.vocab = vocab    # 用于 decode()

    def register_special_tokens(self, special_tokens):
        """
        注册特殊 token。

        特殊 token 是不进行 BPE 编码的，它们直接映射为对应的 token ID。
        例如 GPT-4 中的 <|endoftext|>、<|fim_prefix|> 等。

        参数:
            special_tokens: 字典，格式为 {token字符串: token_id}
                           示例: {'<|endoftext|>': 100257}
        """
        # 设置特殊 token 映射
        self.special_tokens = special_tokens
        # 同时构建反向映射（token ID -> token字符串）
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        将 token ID 列表解码回字符串。

        参数:
            ids: token ID 列表

        返回:
            解码后的字符串

        处理逻辑:
        - 如果 token ID 在 vocab 中，将其转换为字节并拼接
        - 如果 token ID 在特殊 token 反向映射中，直接使用其原始字符串
        - 否则抛出错误
        """
        part_bytes = []  # 存储解码后的字节片段
        for idx in ids:
            if idx in self.vocab:
                # 普通 token：从 vocab 获取字节序列
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # 特殊 token：直接编码为 UTF-8 字节
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"无效的 token id: {idx}")

        # 拼接所有字节片段
        text_bytes = b"".join(part_bytes)
        # 解码为字符串
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """
        对一个字节块进行 BPE 编码。

        这是编码的核心逻辑，对单个文本块（已经转换为字节）应用 BPE 合并规则。

        参数:
            text_bytes: 字节序列（bytes 对象）

        返回:
            token ID 列表

        算法:
        1. 将字节转换为整数列表
        2. 迭代查找可合并的字节对：
           - 找到合并索引最小的字节对（优先合并先出现的对）
        3. 当没有可合并的对时停止
        """
        # 将所有字节转换为 0-255 的整数
        ids = list(text_bytes)

        # 迭代合并，只要列表长度 >= 2 就继续
        while len(ids) >= 2:
            # 统计所有字节对的频率
            stats = get_stats(ids)

            # 找到合并索引最小的字节对
            # 使用 float("inf") 作为不存在合并时的默认值
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # 细节：如果没有更多的合并可用，key 会返回 inf，
            # min 会任意选择列表中的第一个对
            # 我们通过检查 pair 是否在 merges 中来检测这种终止情况
            if pair not in self.merges:
                break  # 无法再合并，退出循环

            # 执行合并：用新 token ID 替换该字节对
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def encode_ordinary(self, text):
        """
        普通编码（忽略所有特殊 token）。

        参数:
            text: 输入字符串

        返回:
            token ID 列表

        工作流程:
        1. 使用正则模式将文本分割成文本块
        2. 对每个文本块调用 _encode_chunk 进行 BPE 编码
        3. 拼接所有块的编码结果
        """
        # 使用正则模式分割文本为文本块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 对每个文本块分别编码，然后拼接结果
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # 转换为原始字节
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)  # 拼接编码结果

        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        编码文本，处理特殊 token。

        参数:
            text: 输入字符串
            allowed_special: 特殊 token 处理方式
                           - "all": 编码所有特殊 token
                           - "none": 不处理特殊 token，当普通文本编码
                           - "none_raise": 如果遇到特殊 token 则报错（默认）
                           - set: 只编码指定的特殊 token

        返回:
            token ID 列表

        工作流程:
        1. 根据 allowed_special 确定要处理的特殊 token 集合
        2. 使用正则表达式分割文本，分离普通文本和特殊 token
        3. 对每个部分分别编码（特殊 token 直接映射，普通文本调用 encode_ordinary）
        4. 拼接编码结果

        注意:
        这是 tiktoken 的默认行为，遇到未知特殊 token 时报错，
        可以避免很多潜在问题。
        """
        # 解码用户对特殊 token 处理的期望
        special = None
        if allowed_special == "all":
            # 处理所有特殊 token
            special = self.special_tokens
        elif allowed_special == "none":
            # 不处理特殊 token
            special = {}
        elif allowed_special == "none_raise":
            # 如果文本中包含特殊 token 则报错
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            # 只处理指定集合中的特殊 token
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} 不支持")

        # 如果没有特殊 token，直接使用普通编码
        if not special:
            return self.encode_ordinary(text)

        # 否则需要仔细处理可能存在的特殊 token
        # 使用正则表达式基于特殊 token 分割文本
        # 将模式用括号包围使其成为捕获组，这样特殊 token 会被包含在结果中
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        # 现在所有特殊字符都与普通文本分离了
        # 对每个文本块分别编码，然后拼接结果
        ids = []
        for part in special_chunks:
            if part in special:
                # 这是特殊 token，作为特殊情况单独编码
                ids.append(special[part])
            else:
                # 这是普通文本，使用普通编码
                ids.extend(self.encode_ordinary(part))

        return ids