"""
minbpe - 最简 BPE (Byte Pair Encoding) 算法实现

一个轻量级的 BPE Tokenizer 库，包含三个Tokenizer实现：

1. BasicTokenizer - 最简单的 BPE 实现
   - 无正则表达式预处理
   - 无特殊 token 支持

2. RegexTokenizer - 带正则表达式预处理的 BPE
   - 使用正则模式分割文本
   - 支持特殊 token

3. GPT4Tokenizer - GPT-4 Tokenizer 的轻量包装
   - 从 tiktoken 加载预训练的 'cl100k_base' tokenizer
   - 匹配 GPT-4 的编码/解码行为

使用方法:
    from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

    # 创建 tokenizer
    tokenizer = RegexTokenizer()

    # 训练（BasicTokenizer 和 RegexTokenizer 支持）
    tokenizer.train(text, vocab_size=1000)

    # 编码
    tokens = tokenizer.encode("Hello, world!")

    # 解码
    text = tokenizer.decode(tokens)

算法原理:
    BPE 是一种简单的数据压缩算法，通过迭代合并最频繁出现的字符对来构建词汇表。
    1. 将文本分解为字符
    2. 统计所有相邻字符对的频率
    3. 找到最频繁的对，合并为一个新 token
    4. 重复直到达到目标词汇表大小

来源: https://github.com/karpathy/minbpe
"""

from .base import Tokenizer
from .basic import BasicTokenizer
from .regex import RegexTokenizer
from .gpt4 import GPT4Tokenizer