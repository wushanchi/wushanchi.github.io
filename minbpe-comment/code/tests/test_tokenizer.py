"""
minbpe 库的测试文件

该文件包含对所有 Tokenizer 实现的全面测试：
- BasicTokenizer
- RegexTokenizer
- GPT4Tokenizer

测试覆盖:
1. 编码/解码一致性: encode(decode(x)) == x
2. 与官方 GPT-4 tokenizer (tiktoken) 的等效性
3. 特殊 token 的处理
4. Wikipedia BPE 示例验证
5. 保存/加载功能

运行方式:
    pytest                    # 运行所有测试
    pytest -v .              # 详细输出
    pytest tests/test_tokenizer.py  # 直接运行本文件
"""

import pytest
import tiktoken
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# -----------------------------------------------------------------------------
# 公共测试数据

# 用于测试 tokenizers 的几个字符串
test_strings = [
    "",                                   # 空字符串
    "?",                                 # 单个字符
    "hello world!!!? (안녕하세요!) lol123 😉",  # 包含多语言和表情符号的有趣小字符串
    "FILE:taylorswift.txt",               # FILE: 开头的会被特殊处理
]

def unpack(text):
    """
    根据文本内容决定返回什么。

    这是必要的，因为 `pytest -v .` 会打印参数到控制台，
    如果打印整个文件内容会很混乱。所以这里做了一个分支：
    - 如果文本以 "FILE:" 开头，从文件读取内容
    - 否则直接返回文本

    参数:
        text: 文件路径或普通文本

    返回:
        文件内容或原始文本
    """
    if text.startswith("FILE:"):
        # 获取测试文件的目录
        dirname = os.path.dirname(os.path.abspath(__file__))
        # 构建完整文件路径
        taylorswift_file = os.path.join(dirname, text[5:])
        # 读取并返回文件内容
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text


# 用于测试特殊 token 处理的长字符串
specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()

# GPT-4 特殊 token 的官方映射
special_tokens = {
    '<|endoftext|>': 100257,   # 文本结束标记
    '<|fim_prefix|>': 100258,  # FIM 前缀
    '<|fim_middle|>': 100259,   # FIM 中间部分
    '<|fim_suffix|>': 100260,   # FIM 后缀
    '<|endofprompt|>': 100276   # 提示结束标记
}

# 用于测试的长文本（Wikipedia 关于 llama 的文章节选）
llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

# -----------------------------------------------------------------------------
# 测试用例

# 参数化测试：验证所有 tokenizer 的编码/解码一致性
# 测试所有 tokenizer 类型和所有测试字符串的组合
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    """
    测试编码/解码的一致性。

    验证对于各种文本，encode(decode(x)) == x 成立。

    参数化:
        tokenizer_factory: BasicTokenizer, RegexTokenizer, GPT4Tokenizer
        text: 测试字符串列表中的任意一个

    验证:
        1. 解码编码后的 token 列表应该能还原原始文本
    """
    text = unpack(text)  # 如果是 FILE: 开头的，从文件读取
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)  # 编码
    decoded = tokenizer.decode(ids)  # 解码
    assert text == decoded  # 应该完全相等


# 参数化测试：验证 GPT4Tokenizer 与官方 tiktoken 的等效性
@pytest.mark.parametrize("text", test_strings)
def test_gpt4_tiktoken_equality(text):
    """
    测试 GPT4Tokenizer 与官方 GPT-4 tokenizer (tiktoken) 的等效性。

    验证我们的 GPT4Tokenizer 实现与 OpenAI 的 tiktoken 库产生完全相同的结果。

    参数化:
        text: 测试字符串列表中的任意一个

    验证:
        - 我们的 tokenizer 编码结果与 tiktoken 完全相同
    """
    text = unpack(text)
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")  # 官方 tokenizer

    # 使用两种 tokenizer 编码
    tiktoken_ids = enc.encode(text)        # 官方 tiktoken
    gpt4_tokenizer_ids = tokenizer.encode(text)  # 我们的实现

    # 结果应该完全相同
    assert gpt4_tokenizer_ids == tiktoken_ids


# 测试特殊 token 的处理
def test_gpt4_tiktoken_equality_special_tokens():
    """
    测试 GPT4Tokenizer 对特殊 token 的处理与 tiktoken 等效。

    验证当 allowed_special="all" 时，两种 tokenizer 对包含特殊 token 的文本
    编码结果完全相同。
    """
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")

    # 使用 allowed_special="all" 编码（处理所有特殊 token）
    tiktoken_ids = enc.encode(specials_string, allowed_special="all")
    gpt4_tokenizer_ids = tokenizer.encode(specials_string, allowed_special="all")

    assert gpt4_tokenizer_ids == tiktoken_ids


# Wikipedia BPE 示例测试
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    参考测试：验证 BPE 算法的正确性。

    使用 Wikipedia 的 BPE 示例验证算法实现。
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    Wikipedia 示例:
    - 输入字符串: "aaabdaaabac"
    - 执行 3 次合并后的结果: "XdXac"
    - 合并规则:
        Z = aa (出现 2 次)
        Y = ab (出现 2 次)
        X = ZY = aaab

    字符 ASCII 值: a=97, b=98, c=99, d=100

    预期 token ID 序列: [258, 100, 258, 97, 99]
    即: X d X a c
        258 100 258 97 99

    验证:
    1. 编码结果与 Wikipedia 一致
    2. decode(encode(x)) == x
    """
    tokenizer = tokenizer_factory()
    text = "aaabdaaabac"

    # 训练 vocab_size = 256 + 3 = 259（初始 256 字节 + 3 次合并）
    tokenizer.train(text, 256 + 3)

    ids = tokenizer.encode(text)

    # 验证编码结果
    assert ids == [258, 100, 258, 97, 99]

    # 验证解码一致性
    assert tokenizer.decode(tokenizer.encode(text)) == text


# 保存/加载功能测试
@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load(special_tokens):
    """
    测试 tokenizer 的保存和加载功能。

    验证:
    1. 训练后的 tokenizer 可以保存到文件
    2. 保存后可以重新加载
    3. 加载后的 tokenizer 与原始 tokenizer 功能完全相同
    4. 特殊 token 可以正确注册和恢复

    参数化:
        special_tokens: 空字典 {} 或包含特殊 token 的字典
    """
    # 使用 llama_text（较复杂的文本）训练 tokenizer，执行 64 次合并
    text = llama_text

    # 创建 RegexTokenizer 并训练
    tokenizer = RegexTokenizer()
    tokenizer.train(text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)

    # 验证 decode(encode(x)) == x
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text

    # 保存 tokenizer
    ids = tokenizer.encode(text, "all")
    tokenizer.save("test_tokenizer_tmp")  # 会创建 .model 和 .vocab 文件

    # 重新加载 tokenizer
    tokenizer = RegexTokenizer()
    tokenizer.load("test_tokenizer_tmp.model")

    # 验证加载后的功能
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    assert tokenizer.encode(text, "all") == ids

    # 删除临时文件
    for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
        os.remove(file)


# 运行测试的入口点
if __name__ == "__main__":
    pytest.main()