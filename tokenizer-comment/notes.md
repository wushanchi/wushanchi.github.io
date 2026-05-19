# GPT Tokenizer — BPE 算法详解

> 本课程讲解 Byte Pair Encoding (BPE) 算法和 Tokenizer 实现。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) |
| **源代码** | [karpathy/minbpe](https://github.com/karpathy/minbpe) |
| **核心主题** | BPE 算法、Tokenization、GPT Tokenizer |

---

## 🎯 学习目标

1. 理解 BPE 算法的原理
2. 掌握 Tokenizer 的实现
3. 理解 Tokenization 的问题
4. 了解为什么理想情况下应删除此阶段

---

## 🧠 BPE 算法

### 核心思想

BPE 是一种简单的数据压缩算法，用于将频繁出现的字节对合并为单个字节。

### 算法步骤

```
1. 将文本分解为字节序列
2. 统计所有相邻字节对的出现频率
3. 找到最频繁的字节对 (A, B)
4. 将所有 (A, B) 替换为新的字节 AB
5. 重复步骤 2-4，直到达到预设的词汇表大小
```

### 示例

```
原始: "aaabbc"
字节: [97, 97, 97, 98, 98, 99]

频率统计:
- (97, 97): 2
- (97, 98): 1
- (98, 98): 1
- (98, 99): 1

最频繁: (97, 97) → 合并为 256 (新字节)

新序列: [256, 97, 98, 98, 99]
```

---

## 🔧 Tokenizer 实现

### 基本接口

```python
class Tokenizer:
    def encode(self, text) -> List[int]:
        """将文本转换为 token 序列"""
        pass

    def decode(self, tokens) -> str:
        """将 token 序列转换回文本"""
        pass
```

### 维度说明

```
输入文本: "Hello, world!"
  ↓
encode()
  ↓
输出: [15496, 11, 1917, 0]  (token IDs)
  ↓
decode()
  ↓
输出: "Hello, world!"
```

---

## ⚠️ Tokenization 的问题

### 1. 词汇表大小

- GPT-2 使用 50,000 个 merge
- 导致词汇表为 50,257（加上特殊 token）

### 2. 分布不均匀

- 英文文本效率高（每个 token 约 4 字符）
- 中文/代码效率低

### 3. OOV 问题

- 未登录词（Out-of-Vocabulary）
- 字节级 fallback 可以缓解但不完全解决

---

## 📊 常见 Tokenizer 对比

| Tokenizer | 词汇表大小 | 语言 | 特点 |
|-----------|-----------|------|------|
| GPT-2 BPE | 50,257 | 多语言 | 字节级 |
| SentencePiece | 32k-64k | 多语言 | 词级别 |
| Tiktoken | 100k+ | 英语 | 高效 |

---

## 🚀 使用示例

```python
from minbpe import BasicTokenizer, RegexTokenizer

# 加载 GPT-2 tokenizer
tokenizer = RegexTokenizer()
tokenizer.load("tokenizer.json")

# 编码
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [15496, 11, 1917, 0]

# 解码
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

---

## 🔮 理想情况

Karpathy 指出：理想情况下应该**删除 Tokenizer 阶段**。

原因：
1. Tokenizer 引入额外的复杂性
2. 可能成为模型能力的瓶颈
3. 字节级模型可以避免这个问题

但目前实际应用中，Tokenization 仍然是必要的优化。

---

> 📚 视频: [GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
> 📦 代码: [karpathy/minbpe](https://github.com/karpathy/minbpe)