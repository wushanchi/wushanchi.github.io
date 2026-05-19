# LLM.C — 纯 C/CUDA 实现 LLM 训练

> LLM.C 是 Andrej Karpathy 的纯 C 和 CUDA 实现的大语言模型训练项目，无需任何依赖。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/llm.c](https://github.com/karpathy/llm.c) |
| **核心主题** | 纯 C 训练、CUDA 加速、零依赖 |

---

## 🎯 学习目标

1. 理解底层矩阵运算和 GPU 并行编程
2. 掌握 CUDA 优化技术
3. 学习 LLM 训练的核心组件
4. 理解极致性能优化的方法

---

## 📁 文件结构

```
llm-c-comment/code/
├── train_gpt2.c         # CPU 训练代码
├── train_gpt2.cu        # CUDA 训练代码（主要）
├── train_gpt2_fp32.cu   # FP32 CUDA 版本
├── test_gpt2.c          # CPU 测试代码
├── test_gpt2.cu         # CUDA 测试代码
├── profile_gpt2.cu       # 性能分析代码
├── train_llama3.py      # Python Llama3 训练脚本
├── llmc/                # C 库头文件
│   └── utils.h         # 工具函数
├── .gitignore          # Git 忽略配置
├── Makefile            # 编译配置
├── README.md           # 官方 README
└── requirements.txt    # Python 依赖
```

---

## 🧠 为什么用 C/CUDA？

### 优势

| 优势 | 说明 |
|------|------|
| 零依赖 | 不需要 Python、PyTorch、CUDA 库 |
| 极致性能 | 避免 Python 运行时开销 |
| 易于部署 | 编译后直接运行 |
| 学习底层 | 深入理解 GPU 和矩阵运算 |

### CUDA 优化技术

```
1. 合并内存访问 (Coalesced Memory Access)
   - 相邻线程访问相邻内存地址

2. 共享内存 (Shared Memory)
   - 利用 GPU 片上高速缓存
   - 减少全局内存访问

3. 矩阵分块 (Tiling)
   - 将大矩阵划分为小块
   - 提高缓存命中率

4. 线程束优化 (Warp Optimization)
   - 减少线程束分化
   - 使用 warp-level 原语
```

---

## 🔧 核心组件

### 矩阵乘法 (GEMM)

```c
// 每个线程计算一个输出元素
__global__ void matmul_kernel(
    float* C, float* A, float* B, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### Softmax

```c
// 数值稳定的 softmax
__global__ void softmax_kernel(float* out, float* in, int n) {
    // 1. 找最大值（避免数值溢出）
    // 2. 计算指数和
    // 3. 归一化
}
```

---

## 🔄 训练流程

```
1. 数据加载
   └── 从二进制文件读取 token 序列

2. 模型初始化
   └── 权重随机初始化

3. 训练循环
   ├── 前向传播 (矩阵乘法 + LayerNorm + Softmax)
   ├── 计算损失 (Cross Entropy)
   ├── 反向传播 (梯度计算)
   └── 参数更新 (AdamW)

4. 模型保存
   └── 二进制格式输出
```

---

## 📊 性能对比

| 实现 | 速度 | 说明 |
|------|------|------|
| PyTorch (GPU) | 1x | 基准 |
| LLM.C (CUDA) | ~0.8x | 接近 PyTorch |
| LLM.C (CPU) | ~0.1x | 比 PyTorch 慢 |

---

## 🔨 编译与运行

```bash
# CUDA 编译
nvcc -O3 -o train train_gpt2.cu -lm

# CPU 编译
gcc -O3 -o train train_gpt2.c -lm

# 运行
./train
```

---

## 📚 相关资源

- [karpathy/llm.c GitHub](https://github.com/karpathy/llm.c)
- [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)