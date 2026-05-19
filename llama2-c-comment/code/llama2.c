"""
Llama2.C — 单文件纯 C 推理 Llama 2

来源: https://github.com/karpathy/llama2.c

============ 与 GPT-2 的区别 ============

位置编码: RoPE (Rotary)
FFN: SwiGLU 激活
归一化: RMSNorm
注意力: MQA (Multi-Query Attention)

============ 文件结构 ============

run.c     — 单文件完整推理代码
run.ipynb — Jupyter Notebook 版本
model.py  — Python 参考实现

============ 使用方法 ============

# 编译
gcc -O3 -o run run.c

# 运行
./run prompt.txt
"""

// 以下是 C 代码示例框架

/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SEQ_LEN 2048
#define MAX_VOCAB 32000

// 模型配置
typedef struct {
    int vocab_size;
    int dim;
    int n_layers;
} Config;

// Transformer
typedef struct {
    float* token_embedding;  // [vocab_size, dim]
    float* rmsnorm_weight;  // [dim]
    // ... layers
} Transformer;

// RoPE 位置编码
void rotary(float* q, float* k, int head_dim, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        float theta = i / powf(10000.0, 2.0 * i / head_dim);
        float c = cosf(theta);
        float s = sinf(theta);
        // 应用旋转
        float q0 = q[2*i];
        float q1 = q[2*i+1];
        q[2*i] = q0 * c - q1 * s;
        q[2*i+1] = q0 * s + q1 * c;
    }
}

// SwiGLU 激活
float swiglu(float x) {
    return x * (1.0 / (1.0 + expf(-x)));
}
*/

// 完整代码见: https://github.com/karpathy/llama2.c