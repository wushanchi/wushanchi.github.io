"""
训练脚本 - 演示 minbpe Tokenizer 的训练过程

这个脚本展示了如何在实际文本上训练 BasicTokenizer 和 RegexTokenizer。

运行时间:
    在标准笔记本电脑上约需 25 秒

输出:
    - models/basic.model
    - models/basic.vocab
    - models/regex.model
    - models/regex.vocab
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

# 打开文本文件并训练 512 个 token 的词汇表
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# 创建 models 目录，避免污染当前目录
os.makedirs("models", exist_ok=True)

# 记录开始时间
t0 = time.time()

# 依次训练 BasicTokenizer 和 RegexTokenizer
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    # 构建 Tokenizer 对象并开始详细训练
    tokenizer = TokenizerClass()

    # 训练 vocab_size=512（256 基础字节 + 256 合并 token）
    tokenizer.train(text, 512, verbose=True)

    # 写入两个文件到 models 目录:
    # - name.model: 用于加载 tokenizer（关键文件）
    # - name.vocab: 供人类查看的词汇表
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)

# 记录结束时间并打印训练耗时
t1 = time.time()
print(f"训练耗时: {t1 - t0:.2f} 秒")