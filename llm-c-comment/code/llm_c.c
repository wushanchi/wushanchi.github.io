"""
LLM.C — 纯 C/CUDA 实现 LLM 训练

来源: https://github.com/karpathy/llm.c

============ 核心概念 ============

【为什么用 C？】
- 零依赖（不需要 Python/PyTorch）
- 极致性能
- 易于部署

【CUDA 优化】
- 矩阵乘法 kernel
- 共享内存
- 合并内存访问

============ 文件结构 ============

train_gpt2.cu — CUDA 训练代码
train_gpt2.c   — CPU 训练代码
test_gpt2.cu   — CUDA 测试代码
test_gpt2.c    — CPU 测试代码

============ 编译 ============

# CUDA 编译
nvcc -O3 -o train train_gpt2.cu

# CPU 编译
gcc -O3 -o train train_gpt2.c -lm
"""

// 以下是 C 代码示例框架

/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 矩阵乘法
void matmul(float* C, float* A, float* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

// Softmax
void softmax(float* out, float* in, int n) {
    float max = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max) max = in[i];
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += exp(in[i] - max);
    }
    for (int i = 0; i < n; i++) {
        out[i] = exp(in[i] - max) / sum;
    }
}
*/

// 完整代码见: https://github.com/karpathy/llm.c