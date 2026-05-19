"""
GPT Tokenizer — BPE 算法实现

来源: https://github.com/karpathy/minbpe
视频: https://www.youtube.com/watch?v=zduSFxRajkE

============ BPE 算法步骤 ============

1. 将文本分解为字节序列
2. 统计相邻字节对频率
3. 合并最频繁的对
4. 重复直到达到词汇表大小

============ Tokenizer 接口 ============

class Tokenizer:
    def encode(self, text) -> List[int]:
        """将文本转换为 token 序列"""
        pass

    def decode(self, tokens) -> str:
        """将 token 序列转换回文本"""
        pass

============ GPT-2 Tokenizer 配置 ============

vocab_size = 50257
  - 256 字节
  - 50,000 BPE merges
  - 1 特殊令牌 (<|endoftext|>)
"""

import re
from typing import List


class BasicTokenizer:
    """
    最简单的 BPE Tokenizer 实现

    功能:
    1. 训练: 从文本学习 BPE 合并规则
    2. 编码: 将文本转换为 token IDs
    3. 解码: 将 token IDs 转换回文本
    """

    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.vocab_size = 256  # 初始: 256 字节

    def train(self, text: str, vocab_size: int = 1000):
        """
        训练 BPE Tokenizer

        Args:
            text: 训练文本
            vocab_size: 目标词汇表大小

        复杂度: O(n * vocab_size)
        """
        # 将文本分解为字节
        tokens = list(text.encode('utf-8'))

        # 统计相邻对的频率
        while self.vocab_size < vocab_size:
            # 统计所有相邻对的频率
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # 找到最频繁的对
            best_pair = max(pairs, key=pairs.get)

            # 合并
            new_token = self.vocab_size
            self.merges[best_pair] = new_token

            # 更新 tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token IDs

        Args:
            text: 输入文本

        Returns:
            token IDs 列表
        """
        # 字节级编码
        tokens = list(text.encode('utf-8'))

        # 贪婪应用 BPE 合并
        while True:
            # 找到所有可合并的对
            merges_found = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    merges_found.append((i, self.merges[pair]))

            if not merges_found:
                break

            # 贪婪合并第一个
            i, new_token = merges_found[0]
            tokens = tokens[:i] + [new_token] + tokens[i + 2:]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        将 token IDs 解码为文本

        Args:
            tokens: token IDs 列表

        Returns:
            解码后的文本
        """
        # 反向应用合并
        while True:
            found = False
            for (pair, new_token) in self.merges.items():
                if new_token in tokens:
                    idx = tokens.index(new_token)
                    tokens = tokens[:idx] + list(pair) + tokens[idx + 1:]
                    found = True
                    break

            if not found:
                break

        return bytes(tokens).decode('utf-8', errors='replace')


class RegexTokenizer:
    """
    带正则化预处理的 BPE Tokenizer

    在编码前使用正则表达式预处理文本:
    - 分离标点符号
    - 分割数字
    - 处理空白
    """

    def __init__(self):
        self.basic = BasicTokenizer()
        self.pattern = re.compile(r"""(?:[^\s\w]|_)+|(?:\d+(?:\.\d*)?)|(?:\d*(?:\.\d+)?)""")

    def train(self, text: str, vocab_size: int = 1000):
        # 预处理文本
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        self.basic.train(text, vocab_size)

    def encode(self, text: str) -> List[int]:
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        return self.basic.encode(text)

    def decode(self, tokens: List[int]) -> str:
        text = self.basic.decode(tokens)
        return text.replace(' ', '')


# 示例
if __name__ == '__main__':
    text = "Hello, world! 123"

    tokenizer = RegexTokenizer()
    tokenizer.train("Hello world " * 100, vocab_size=500)

    tokens = tokenizer.encode("Hello, world! 123")
    print(f"Encoded: {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")