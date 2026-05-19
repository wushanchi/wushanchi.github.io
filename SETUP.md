# Python 环境配置指南

> 本指南帮助您配置 Python 环境以运行本仓库中的所有代码

---

## 📋 目录

1. [环境要求](#环境要求)
2. [安装 Python](#安装-python)
3. [创建虚拟环境](#创建虚拟环境)
4. [安装依赖](#安装依赖)
5. [验证安装](#验证安装)
6. [运行代码](#运行代码)
7. [常见问题](#常见问题)

---

## 🔧 环境要求

### 最低要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.8 |
| pip | >= 21.0 |
| 内存 | >= 4GB |
| 磁盘 | >= 2GB |

### 推荐配置

| 项目 | 推荐 |
|------|------|
| Python | 3.10 或 3.11 |
| pip | 最新版本 |
| 显卡 | NVIDIA GPU (用于 GPU 训练) |
| CUDA | 11.8+ (如使用 GPU) |

---

## 📥 安装 Python

### macOS

```bash
# 使用 Homebrew 安装
brew install python@3.10

# 验证安装
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
# 更新包列表
sudo apt update

# 安装 Python 和 pip
sudo apt install python3.10 python3-pip

# 验证安装
python3 --version
```

### Windows

1. 下载 [Python 3.10+](https://www.python.org/downloads/)
2. 运行安装程序
3. **重要**: 勾选 "Add Python to PATH"
4. 点击 "Install Now"

---

## 🧪 创建虚拟环境

### 为什么需要虚拟环境？

虚拟环境可以隔离不同项目的依赖，避免版本冲突。

### 使用 venv（推荐）

```bash
# 进入项目目录
cd karpathy-gpt-roadmap

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境

# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 使用 conda

```bash
# 创建 conda 环境
conda create -n karpathy python=3.10

# 激活环境
conda activate karpathy
```

---

## 📦 安装依赖

### micrograd（反向传播）

```bash
# 激活虚拟环境后
pip install torch

# 验证安装
python -c "import torch; print(torch.__version__)"
```

### makemore（语言模型）

```bash
# 安装 PyTorch
pip install torch

# 安装数据处理和可视化依赖
pip install numpy matplotlib

# 验证
python -c "import torch; import numpy; print('OK')"
```

### nanogpt-build（GPT 训练）

```bash
# 安装 PyTorch（GPU 版本）
pip install torch torchvision torchaudio

# 安装训练辅助工具
pip install tensorboard tqdm

# 验证 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 完整安装（一次性安装所有）

```bash
# 创建 requirements.txt 在项目根目录
cat > requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
tqdm>=4.65.0
EOF

# 安装所有依赖
pip install -r requirements.txt
```

---

## ✅ 验证安装

### 运行以下测试脚本

```python
# test_environment.py

import sys

def test_imports():
    """测试所有必要的包是否可导入"""
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
    }

    print("=" * 50)
    print("Python 环境验证")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print()

    all_ok = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")
            all_ok = False

    print()
    if all_ok:
        print("✓ 所有依赖已正确安装!")
    else:
        print("✗ 部分依赖缺失，请重新安装")

    return all_ok

def test_gpu():
    """测试 GPU 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ GPU 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("\n⚠ GPU 不可用，将使用 CPU 训练（速度较慢）")
    except Exception as e:
        print(f"\n⚠ GPU 检测失败: {e}")

if __name__ == '__main__':
    test_imports()
    test_gpu()
```

运行测试：

```bash
python test_environment.py
```

预期输出：

```
==================================================
Python 环境验证
==================================================
Python 版本: 3.10.12

✓ PyTorch: 2.0.1
✓ NumPy: 1.24.3
✓ Matplotlib: 3.7.1

✓ 所有依赖已正确安装!

✓ GPU 可用: NVIDIA GeForce RTX 3080
  CUDA 版本: 11.8
```

---

## 🚀 运行代码

### 1. micrograd（反向传播基础）

```bash
# 进入 micrograd 目录
cd micrograd-comment

# 运行示例
python -c "
from micrograd.engine import Value

# 创建变量
a = Value(2.0)
b = Value(3.0)

# 计算
c = a * b + a

# 反向传播
c.backward()

print(f'a.grad = {a.grad}')  # b + 1 = 4
print(f'b.grad = {b.grad}')  # a = 2
"
```

### 2. makemore（字符级语言模型）

```bash
# 假设已有 names.txt 数据文件
python makemore_part1.py --input-file names.txt --type bigram --max-steps 100

# 采样生成
python makemore_part1.py --sample-only --type bigram
```

### 3. nanogpt-build（GPT 训练）

```bash
# 训练 GPT-2
python train_gpt2.py --input-file data.txt

# 使用预训练模型采样
python train_gpt2.py --sample-only --checkpoint out/model.pt
```

---

## ❓ 常见问题

### Q1: pip 安装失败

**错误**: `pip: command not found` 或 `ModuleNotFoundError: No module named 'pip'`

**解决方案**:

```bash
# Linux/macOS
python3 -m ensurepip --upgrade

# 或使用 get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

### Q2: CUDA 不可用

**检查步骤**:

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")

# 如果 CUDA 不可用但想使用 GPU
# 访问 https://pytorch.org/get-started/locally/
# 选择正确的 CUDA 版本安装
```

### Q3: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:

1. 减小 batch_size
2. 使用更小的模型
3. 清理 GPU 缓存：

```python
import torch
torch.cuda.empty_cache()
```

### Q4: 权限错误

**错误**: `Permission denied` 或 `EACCESS`

**解决方案**:

```bash
# 不要使用 sudo pip install
# 使用虚拟环境代替

python3 -m venv myenv
source myenv/bin/activate
pip install package-name
```

### Q5: 包版本冲突

**解决方案**:

```bash
# 卸载所有版本
pip uninstall torch numpy matplotlib -y

# 重新安装兼容版本
pip install torch==2.0.1 numpy==1.24.3 matplotlib==3.7.1
```

---

## 🐳 使用 Docker（可选）

如果您遇到环境配置问题，可以使用 Docker：

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["/bin/bash"]
```

构建和运行：

```bash
# 构建镜像
docker build -t karpathy-learning .

# 运行容器
docker run --gpus all -it karpathy-learning
```

---

## 📚 相关资源

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [Python 虚拟环境教程](https://docs.python.org/3/tutorial/venv.html)
- [CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

> 如遇到其他问题，请在 GitHub 提交 Issue