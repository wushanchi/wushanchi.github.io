"""
nanoGPT 采样脚本

从训练好的模型生成文本样本。支持两种初始化方式：
1. 从本地检查点恢复（init_from='resume'）
2. 从 OpenAI 预训练的 GPT-2 模型加载（init_from='gpt2*'）

使用 tiktoken（OpenAI 的快速 BPE 分词器）对文本进行编码/解码。
"""

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# =============================================================================
# 配置参数
# =============================================================================

# 初始化方式：'resume'（从 out_dir 恢复）或 'gpt2*'（如 'gpt2', 'gpt2-xl'）
init_from = 'resume'

# 输出目录（当 init_from='resume' 时使用）
out_dir = 'out'

# 生成文本的起始提示
start = "\n"  # 也可以使用 "<|endoftext|>" 或其他文本

# 生成样本数量
num_samples = 10

# 每个样本生成的 token 数量
max_new_tokens = 500

# 温度参数：
# 1.0 = 不调整（保持原始分布）
# < 1.0 = 更保守/确定性
# > 1.0 = 更随机/创造性
temperature = 0.8

# Top-K 采样：只保留概率最高的 k 个 token
top_k = 200

# 随机种子
seed = 1337

# 设备：'cpu', 'cuda', 'cuda:0', 'cuda:1' 等，或 Mac 的 'mps'
device = 'cuda'

# 数据类型：优先 bfloat16
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# 是否使用 PyTorch 2.0 编译
compile = False

# =============================================================================
# 配置覆盖（从命令行或配置文件）
# =============================================================================
exec(open('configurator.py').read())

# =============================================================================
# 初始化
# =============================================================================

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 允许 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# =============================================================================
# 模型加载
# =============================================================================

if init_from == 'resume':
    # 从检查点恢复模型
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # 修复可能的前缀问题
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    # 从 OpenAI GPT-2 权重加载
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# 可选：编译模型（需要 PyTorch 2.0）
if compile:
    model = torch.compile(model)

# =============================================================================
# 分词器设置
# =============================================================================

# 尝试从数据集加载 meta.pkl（包含词汇表映射）
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"从 {meta_path} 加载 meta...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # 字符串到索引和索引到字符串的映射
    stoi, itos = meta['stoi'], meta['itos']

    # 编码：字符串 -> token ID 列表
    def encode(s):
        return [stoi[c] for c in s]

    # 解码：token ID 列表 -> 字符串
    def decode(l):
        return ''.join([itos[i] for i in l])

else:
    # 默认使用 GPT-2 的分词器
    print("未找到 meta.pkl，使用 GPT-2 分词器...")
    enc = tiktoken.get_encoding("gpt2")

    def encode(s):
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(l):
        return enc.decode(l)


# =============================================================================
# 文本生成
# =============================================================================

# 处理起始提示（如果是文件则读取内容）
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# 编码起始文本
start_ids = encode(start)

# 转换为张量并添加批次维度：(1, T)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# 生成样本
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # 自回归生成
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            # 解码并打印
            print(decode(y[0].tolist()))
            print('---------------')