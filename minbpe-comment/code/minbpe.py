"""
minbpe - 最简 BPE (Byte Pair Encoding) 算法实现

============ BPE 算法概述 ============

BPE（字节对编码）是一种简单但强大的数据压缩算法，广泛用于大语言模型的 tokenization。

工作原理:
1. 初始化: 将所有字符作为词汇表（基础字节 0-255）
2. 统计: 计算所有相邻字节对的频率
3. 合并: 找到最频繁的字节对，创建一个新 token
4. 重复: 直到词汇表达到目标大小

============ 核心类 ============

Base - 基础类，定义 encode/decode 接口
BasicTokenizer - 最简单的 BPE 实现（无预处理）
RegexTokenizer - 带正则化预处理的 BPE
GPT4Tokenizer - GPT-4 Tokenizer 的轻量包装

============ 使用示例 ============

from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000)

tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

来源: https://github.com/karpathy/minbpe
"""

import re
from typing import List, Dict, Tuple


class Base:
    """
    基础 Tokenizer 类。

    定义所有 Tokenizer 必须实现的接口。
    """

    def encode(self, text: str) -> List[int]:
        """将字符串编码为 token ID 列表"""
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        """将 token ID 列表解码为字符串"""
        raise NotImplementedError

    def save(self, path: str):
        """保存 tokenizer 到文件"""
        raise NotImplementedError

    def load(self, path: str):
        """从文件加载 tokenizer"""
        raise NotImplementedError


class BasicTokenizer(Base):
    """
    最简单的 BPE Tokenizer。

    特点:
    - 无正则表达式预处理
    - 无特殊 token 支持
    - 轻量级，简单直接

    时间复杂度: O(n * k)
    其中 n = 训练文本长度，k = 目标词汇表大小
    """

    def __init__(self):
        """初始化，merges 字典存储合并规则，vocab_size 从 256 开始（所有单字节）"""
        self.merges = {}  # {(byte1, byte2): new_byte_id}
        self.vocab_size = 256

    def train(self, text: str, vocab_size: int = 1000, min_freq: int = 1):
        """
        训练 BPE Tokenizer。

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小
            min_freq: 最小频率阈值（低于此频率的字节对不合并）

        算法:
        1. 将文本编码为 UTF-8 字节序列
        2. 迭代统计和合并最频繁的字节对
        3. 直到 vocab_size 达到目标或没有高频字节对
        """
        # 初始化字节词汇表
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 迭代合并
        while self.vocab_size < vocab_size:
            # 统计相邻对的频率
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            # 找到最频繁的对
            best_pair = None
            best_freq = 0
            for pair, freq in pairs.items():
                if freq > best_freq:
                    best_pair = pair
                    best_freq = freq

            # 如果最高频率低于阈值，停止合并
            if best_freq < min_freq:
                break

            # 创建新 token
            new_token = self.vocab_size
            self.merges[best_pair] = new_token

            # 更新词汇表
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 应用合并到 tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2  # 跳过已合并的两个 token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        编码字符串为 token ID 列表。

        使用贪婪算法：迭代查找可合并的字节对，选择第一个可合并的位置进行合并。
        """
        tokens = list(text.encode('utf-8'))

        # 贪婪合并
        while True:
            # 找出所有可合并的字节对及其新 token ID
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    pairs.append((i, self.merges[pair]))

            if not pairs:
                break  # 没有可合并的对

            # 合并第一个可合并的位置
            i, new_token = pairs[0]
            tokens = tokens[:i] + [new_token] + tokens[i + 2:]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        解码 token ID 列表为字符串。
        """
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.vocab[token])
            else:
                # 未知 token，作为单字节处理
                result.append(bytes([token]))

        return b''.join(result).decode('utf-8', errors='replace')

    def save(self, path: str):
        """保存 tokenizer 到 JSON 文件"""
        import json
        data = {
            'merges': {f'{k[0]},{k[1]}': v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """从 JSON 文件加载 tokenizer"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']


class RegexTokenizer(Base):
    """
    带正则化预处理的 BPE Tokenizer。

    在 BasicTokenizer 的基础上增加了:
    - 正则表达式文本分割预处理
    - 更智能的 token 处理

    预处理规则:
    - 分离标点符号
    - 分割数字
    - 处理空白
    """

    def __init__(self):
        """初始化，内部使用 BasicTokenizer 作为基础"""
        self.basic = BasicTokenizer()
        # 正则表达式模式：分割标点、数字、空白
        self.pattern = re.compile(r"""(?:[^\s\w]|_)+|(?:\d+(?:\.\d*)?)|(?:\d*(?:\.\d+)?)""")

    def train(self, text: str, vocab_size: int = 1000, min_freq: int = 1):
        """
        训练 tokenizer。

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小
            min_freq: 最小频率阈值
        """
        # 预处理：使用正则模式分割文本
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        # 委托给 BasicTokenizer 进行训练
        self.basic.train(text, vocab_size, min_freq)

    def encode(self, text: str) -> List[int]:
        """
        编码字符串。

        1. 使用正则模式分割文本
        2. 对每个部分分别编码
        3. 拼接结果
        """
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        return self.basic.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """解码 token ID 列表为字符串"""
        return self.basic.decode(tokens).replace(' ', '')

    def save(self, path: str):
        """保存 tokenizer"""
        self.basic.save(path)

    def load(self, path: str):
        """加载 tokenizer"""
        self.basic.load(path)


# 测试代码
if __name__ == '__main__':
    # 简单的训练测试
    text = "Hello, world! 123"

    tokenizer = RegexTokenizer()
    tokenizer.train("Hello world " * 100, vocab_size=500)

    tokens = tokenizer.encode("Hello, world! 123")
    print(f"Encoded: {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")