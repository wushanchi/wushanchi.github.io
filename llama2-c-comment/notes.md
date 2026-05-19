# Llama2.C — 单文件纯 C 推理 Llama 2

> Llama2.C 是 Andrej Karpathy 的单文件纯 C 实现的 Llama 2 推理引擎。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/llama2.c](https://github.com/karpathy/llama2.c) |
| **核心主题** | 纯 C 推理、Llama 2 架构、单文件实现 |

---

## 🎯 学习目标

1. 理解 Llama 2 架构与 GPT-2 的区别
2. 掌握纯 C 推理实现
3. 学习 RoPE 位置编码
4. 理解 SwiGLU 激活函数

---

## 📁 文件结构

```
llama2-c-comment/code/
├── run.c                 # 主推理代码（单文件）
├── run.ipynb            # Jupyter Notebook 版本
├── model.py             # Python 参考实现
├── tokenizer.py         # Tokenizer 实现
├── sample.py            # 采样逻辑
├── test.c               # 测试代码
├── test_all.py          # 完整测试
├── train.py             # 训练脚本
├── tinystories.py       # TinyStories 数据处理
├── export.py            # 模型导出
├── configurator.py      # 配置工具
├── Makefile             # 编译配置
└── README.md           # 官方 README
```

---

## 🏗️ Llama 2 架构

### 与 GPT-2 的区别

| 组件 | GPT-2 | Llama 2 |
|------|-------|---------|
| 位置编码 | 固定位置编码 | Rotary (RoPE) |
| FFN 激活 | GELU | SwiGLU |
| 归一化 | Pre-LayerNorm | RMSNorm |
| 注意力 | Multi-Head | Grouped Multi-Query (MQA) |

### RoPE — Rotary Position Embedding

```c
// 旋转位置编码
void rotary(float* q, float* k, int head_dim, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        float theta = i / powf(10000.0, 2.0 * i / head_dim);
        float c = cosf(theta);
        float s = sinf(theta);
        // 应用旋转
        float q0 = q[2*i]; float q1 = q[2*i+1];
        q[2*i] = q0 * c - q1 * s;
        q[2*i+1] = q0 * s + q1 * c;
    }
}
```

### SwiGLU 激活

```c
// SwiGLU = SiLU * Gate
float swiglu(float x) {
    return x / (1.0 + expf(-x));  // SiLU (Sigmoid Linear Unit)
}
```

---

## 🔧 纯 C 实现

### 单文件结构

```c
// run.c - 完整的推理代码
int main(int argc, char* argv[]) {
    // 1. 加载模型
    Transformer transformer = load_model("model.bin");

    // 2. 初始化 tokenizer
    Tokenizer tokenizer = load_tokenizer("tokenizer.bin");

    // 3. 采样循环
    int token = 1;  // <START>
    for (int i = 0; i < max_new_tokens; i++) {
        // 前向传播
        float* logits = forward(&transformer, token);

        // 采样
        token = sample(logits, temperature);

        // 输出
        printf("%s", tokenizer.decode(token));
    }
}
```

---

## 📊 模型文件格式

```
+------------------+
| header (元数据)   |  vocab_size, dim, n_layers, etc.
+------------------+
| embedding table  |  float[vocab_size][dim]
+------------------+
| layers           |  repeated n_layers times:
|  - attention     |    q, k, v, o projections
|  - feedforward   |    gate, up, down projections (SwiGLU)
|  - rmsnorm       |    weights for each layer
+------------------+
| final rmsnorm    |
+------------------+
```

---

## 🚀 编译与运行

```bash
# 编译
gcc -O3 -o run run.c -lm

# 运行
./run prompt.txt
```

---

## 📈 性能

| 平台 | 速度 |
|------|------|
| CPU (单线程) | ~50 tokens/s |
| CPU (多线程) | ~200 tokens/s |
| GPU (CUDA) | ~1000 tokens/s |

---

## 📚 相关资源

- [karpathy/llama2.c GitHub](https://github.com/karpathy/llama2.c)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202)