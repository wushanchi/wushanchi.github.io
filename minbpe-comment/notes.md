# minbpe — 最简 BPE 算法实现

> minbpe 是 Andrej Karpathy 实现的一个精简 BPE (Byte Pair Encoding) 算法库，用于学习 tokenization 原理。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/minbpe](https://github.com/karpathy/minbpe) |
| **核心主题** | BPE 算法、Tokenization、分词器实现 |

---

## 🎯 学习目标

1. 理解 BPE 算法的原理和实现
2. 掌握 Tokenizer 的构建过程
3. 理解正则化预处理的作用
4. 学习词汇表构建和 token 编码/解码

---

## 📁 文件结构

```
minbpe-comment/code/
├── minbpe/                 # 核心代码包
│   ├── __init__.py        # 包初始化，导出主要类
│   ├── base.py            # 基类 Base 定义 encode/decode 接口
│   ├── basic.py           # BasicTokenizer 最简单 BPE 实现
│   ├── regex.py          # RegexTokenizer 带正则预处理
│   └── gpt4.py           # GPT4Tokenizer 更复杂的实现
├── tests/                 # 测试
│   └── test_tokenizer.py # 单元测试
├── .gitignore            # Git 忽略配置
├── LICENSE               # MIT 许可证
├── README.md             # 官方 README
├── exercise.md           # 练习题
├── lecture.md            # 课程讲义
├── requirements.txt      # Python 依赖
└── train.py             # 训练脚本
```

---

## 🧠 BPE 算法原理

### 算法步骤

```
1. 初始化：将所有字符作为词汇表（256 个字节）
2. 统计：计算所有相邻字节对的频率
3. 合并：找到最频繁的对，添加到词汇表
4. 重复：直到达到目标词汇表大小
```

### 示例

```
原始文本: "aaabbc"
字节序列: [97, 97, 97, 98, 98, 99]

频率统计:
- (97, 97) = 2 次  ← 最频繁
- (97, 98) = 1 次
- (98, 98) = 1 次
- (98, 99) = 1 次

合并 (97, 97) → 256
新序列: [256, 97, 98, 98, 99]
```

---

## 🔧 核心类

### Base — 抽象基类

```python
class Base:
    def train(self, text, vocab_size): ...
    def encode(self, text) -> List[int]: ...
    def decode(self, tokens) -> str: ...
    def save(self, path): ...
    def load(self, path): ...
```

### BasicTokenizer — 简单 BPE

```python
class BasicTokenizer(Base):
    def __init__(self):
        self.merges = {}  # 合并规则
        self.vocab = {}  # token → bytes
        self.vocab_size = 256
```

### RegexTokenizer — 正则预处理

```python
class RegexTokenizer(Base):
    def __init__(self):
        self.pattern = re.compile(r"""...""")
        self.basic = BasicTokenizer()
```

---

## 📊 训练复杂度

| 指标 | 复杂度 |
|------|--------|
| 时间 | O(n × k) |
| 空间 | O(n) |

其中：
- n = 训练文本长度
- k = 目标词汇表大小

优化：使用堆（优先队列）可加速到 O(n × log k)

---

## 🔄 编码/解码流程

```
编码流程:
1. 文本 → 正则预处理（分离标点、数字）
2. 字节序列 → 贪心合并（应用所有 merge 规则）
3. 输出 token IDs

解码流程:
1. token IDs → 字节序列（查表）
2. 字节序列 → UTF-8 文本
```

---

## 🧪 运行测试

```bash
cd minbpe-comment/code
python -m pytest tests/test_tokenizer.py
# 或
python tests/test_tokenizer.py
```

---

## 📚 相关资源

- [minbpe GitHub](https://github.com/karpathy/minbpe)
- [BPE 原始论文](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer 原理](https://www.youtube.com/watch?v=zduSFxRajkE)