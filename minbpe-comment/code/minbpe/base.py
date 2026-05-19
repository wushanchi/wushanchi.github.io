"""
包含基础 Tokenizer 类和一些通用的辅助函数。
基类还包含（通用的）保存/加载功能。

严格来说，可以将所有正则/模式相关部分隔离到 RegexTokenizer 中，
但为了简洁做了一些妥协。
"""
import unicodedata

# -----------------------------------------------------------------------------
# 几个对 BasicTokenizer 和 RegexTokenizer 都有用的辅助函数

def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回连续字节对出现次数的字典。

    参数:
        ids: 整数列表（如字节序列的整数表示）
        counts: 可选的现有计数器字典，用于累加统计

    返回:
        字典，键为 (字节1, 字节2) 元组，值为出现次数

    示例:
        [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    注意:
        这个函数使用 zip(ids, ids[1:]) 来获取所有相邻对，
        即 ids[i] 和 ids[i+1] 组成的对。
    """
    counts = {} if counts is None else counts
    # zip 会生成所有相邻元素对: (ids[0], ids[1]), (ids[1], ids[2]), ...
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    在整数列表 (ids) 中，将所有连续出现的 pair 替换为新的整数 token idx。

    参数:
        ids: 整数列表
        pair: 要替换的字节对，如 (1, 2)
        idx: 新的合并后的 token ID

    返回:
        替换后的新列表

    示例:
        ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]

    算法:
        遍历列表，当遇到连续的元素匹配 pair 时，用 idx 替换这两个元素。
        否则保留原元素。
    """
    newids = []
    i = 0
    while i < len(ids):
        # 检查当前元素是否匹配 pair 的第一个元素，
        # 且不是最后一个元素（需要配对元素），
        # 且下一个元素匹配 pair 的第二个元素
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)  # 替换为合并后的 token
            i += 2  # 跳过这两个元素（已被合并）
        else:
            newids.append(ids[i])  # 保留原元素
            i += 1
    return newids


def replace_control_characters(s: str) -> str:
    """
    替换字符串中的控制字符，防止它们在输出时造成混乱。

    控制字符（如 \n、\r 等）会在打印时产生格式问题，
    或者在终端上产生不可见的效果。这个函数将所有控制字符
    转换为可读的转义序列。

    参数:
        s: 输入字符串

    返回:
        字符串，其中控制字符被替换为 \\uXXXX 格式

    参考:
        https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars = []
    for ch in s:
        # Unicode 类别以 "C" 开头的都是控制字符
        # (C0 controls, C1 controls, format characters, etc.)
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # 正常字符，保留
        else:
            # 控制字符转换为转义序列，如 \u0000
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    美化打印一个 token，对控制字符进行转义。

    参数:
        t: token 的字节串

    返回:
        可读的字符串表示，控制字符被转义
    """
    s = t.decode('utf-8', errors='replace')  # 解码，错误用替换字符
    s = replace_control_characters(s)  # 转义控制字符
    return s


# -----------------------------------------------------------------------------
# 基础 Tokenizer 类

class Tokenizer:
    """
    Tokenizer 的基类。

    定义了所有 Tokenizer 必须实现的接口：
    - train(): 从文本训练词汇表
    - encode(): 将字符串编码为 token ID 列表
    - decode(): 将 token ID 列表解码为字符串

    同时提供了 save()/load() 方法用于持久化 tokenizer。
    """

    def __init__(self):
        """
        初始化 Tokenizer。

        默认设置：
        - vocab 大小为 256（所有单字节）
        - 无合并规则
        - 无正则模式
        - 无特殊 token
        """
        self.merges = {}         # (int, int) -> int, 合并规则: (byte1, byte2) -> new_byte_id
        self.pattern = ""         # str, 正则分割模式
        self.special_tokens = {}  # str -> int, 特殊 token 映射，如 {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes, token ID 到字节序列的映射

    def train(self, text, vocab_size, verbose=False):
        """
        训练 tokenizer 从文本构建大小为 vocab_size 的词汇表。

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小（必须 >= 256，因为 0-255 保留给单字节）
            verbose: 是否打印详细训练信息

        注意:
            子类必须实现此方法
        """
        raise NotImplementedError

    def encode(self, text):
        """
        将字符串编码为 token ID 列表。

        参数:
            text: 输入字符串

        返回:
            整数列表，表示 token ID

        注意:
            子类必须实现此方法
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        将 token ID 列表解码为字符串。

        参数:
            ids: token ID 列表

        返回:
            解码后的字符串

        注意:
            子类必须实现此方法
        """
        raise NotImplementedError

    def _build_vocab(self):
        """
        从合并规则构建词汇表。

        词汇表构建规则：
        1. 初始化为 256 个单字节 (0-255)
        2. 对于每个合并规则，添加新的合并 token
        3. 对于每个特殊 token，添加其编码

        返回:
            字典: token_id -> bytes
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 应用所有合并规则
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 添加特殊 token
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        保存 tokenizer 到两个文件：file_prefix.vocab 和 file_prefix.model

        文件格式：
        - .model 文件: 关键文件，用于 load()，包含版本、模式、特殊 token、合并规则
        - .vocab 文件: 人类可读的词汇表，用于检查

        参数:
            file_prefix: 文件名前缀
        """
        # 写入 model 文件：用于后续 load()
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本号
            f.write("minbpe v1\n")
            # 写入正则模式
            f.write(f"{self.pattern}\n")
            # 写入特殊 token：首先写入数量，然后是每个 token 及其 ID
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 写入合并规则
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # 写入 vocab 文件：供人类查看
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # 注意：很多 token 可能是部分 UTF-8 序列，
                # 无法解码为有效字符串。这里使用 errors='replace'
                # 用替换字符 代替无效字节。
                #这也意味着我们不能使用 .vocab 文件进行 load()，
                # 因为这种解码方式是有损的！
                s = render_token(token)
                # 查找这个 token 的子 token（如果存在）
                if idx in inverted_merges:
                    # 如果这个 token 有子 token，美化显示为合并
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则这是叶子 token，直接打印
                    #（这应该是前 256 个 token，即单字节）
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        加载 tokenizer（save() 的逆操作），仅从 .model 文件。

        参数:
            model_file: .model 文件的路径
        """
        assert model_file.endswith(".model")
        # 读取 model 文件
        merges = {}
        special_tokens = {}
        idx = 256  # 合并 token 从 256 开始
        with open(model_file, 'r', encoding="utf-8") as f:
            # 读取版本号
            version = f.readline().strip()
            assert version == "minbpe v1"
            # 读取正则模式
            self.pattern = f.readline().strip()
            # 读取特殊 token
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取合并规则
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()