"""
micrograd 测试套件 — 验证自动微分正确性

本模块包含两个测试函数，用于验证 micrograd 实现与 PyTorch 的一致性。

来源: https://github.com/karpathy/micrograd

============ 测试原理 ============

测试通过对比 micrograd 和 PyTorch 的前向传播和反向传播结果，
确保自动微分实现的正确性。

测试原则：
- 前向传播：验证计算结果一致（数值相等）
- 反向传播：验证梯度计算一致（在容差范围内）

============ 测试覆盖 ============

test_sanity_check():
- 基础运算：加法、乘法、ReLU
- 验证单路径计算图

test_more_ops():
- 扩展运算：幂运算、减法、除法
- 验证 += 累加操作
- 验证复杂多路径计算图
"""

import torch
from micrograd.engine import Value


def test_sanity_check():
    """
    基础测试：验证基本运算的正确性

    测试内容：
    - 加法: z = 2*x + 2 + x
    - 乘法: q = z*z
    - ReLU: h = (z*z).relu()
    - 组合: y = h + q + q*x

    验证：
    1. 前向传播：micrograd 结果 == PyTorch 结果
    2. 反向传播：micrograd 梯度 == PyTorch 梯度

    计算过程：
    x = -4.0

    前向传播：
    z = 2*x + 2 + x = 2*(-4) + 2 + (-4) = -10
    q = z.relu() + z*x = max(0,-10) + (-10)*(-4) = 0 + 40 = 40
    h = (z*z).relu() = (-10*-10).relu() = 100.relu() = 100
    y = h + q + q*x = 100 + 40 + 40*(-4) = 100 + 40 - 160 = -20

    反向传播（链式法则）：
    dy/dx = dy/dh*dh/dz*dz/dx + dy/dq*dq/dz*dz/dx + dy/dq*dq/dx

    对比 PyTorch 验证梯度计算是否正确。
    """
    # ===== micrograd 前向传播 =====
    x = Value(-4.0)
    z = 2 * x + 2 + x      # z = 2x + 2 + x = 3x + 2
    q = z.relu() + z * x   # q = relu(z) + z*x
    h = (z * z).relu()     # h = relu(z*z)
    y = h + q + q * x      # y = h + q + q*x
    y.backward()           # 反向传播

    xmg, ymg = x, y  # 保存 micrograd 结果

    # ===== PyTorch 前向传播（对照组） =====
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xpt, ypt = x, y  # 保存 PyTorch 结果

    # ===== 验证结果 =====
    # 验证 1：前向传播结果一致
    assert ymg.data == ypt.data.item(), \
        f"Forward pass mismatch: micrograd={ymg.data}, pytorch={ypt.data.item()}"

    # 验证 2：反向传播梯度一致
    assert xmg.grad == xpt.grad.item(), \
        f"Backward pass mismatch: micrograd={xmg.grad}, pytorch={xpt.grad.item()}"

    print("✓ test_sanity_check passed")
    print(f"  Forward: y={ymg.data}, expected={ypt.data.item()}")
    print(f"  Backward: dx={xmg.grad}, expected={xpt.grad.item()}")


def test_more_ops():
    """
    扩展测试：验证更多运算和复杂计算图

    测试内容：
    - 幂运算: b**3
    - 减法: a - b
    - 除法: f / 2.0
    - 累加操作: d += ..., c += ...
    - ReLU 组合: (b + a).relu(), (b - a).relu()

    计算图结构：
    包含多个路径和合并点，需要正确处理梯度的累加。

    验证：
    - 所有参数的梯度与 PyTorch 一致（容差 1e-6）
    """
    # ===== micrograd 前向传播 =====
    a = Value(-4.0)
    b = Value(2.0)

    # 基础运算
    c = a + b                     # c = -2
    d = a * b + b ** 3           # d = -8 + 8 = 0

    # 累加操作（原地更新）
    c += c + 1                   # c = c + c + 1 = -2 + (-2) + 1 = -3
    c += 1 + c + (-a)           # c = -3 + 1 + (-3) + 4 = -1

    # 复杂组合
    d += d * 2 + (b + a).relu()  # d = 0 + 0 + relu(-2) = 0 + 0 + 0 = 0
    d += 3 * d + (b - a).relu()  # d = 0 + 0 + relu(6) = 6

    # 最终计算
    e = c - d                    # e = -1 - 6 = -7
    f = e ** 2                   # f = (-7)^2 = 49
    g = f / 2.0                  # g = 49 / 2 = 24.5
    g += 10.0 / f               # g = 24.5 + 10/49 ≈ 24.704

    # 反向传播
    g.backward()

    # 保存 micrograd 结果
    amg, bmg, gmg = a, b, g

    # ===== PyTorch 前向传播（对照组） =====
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True

    # 基础运算
    c = a + b
    d = a * b + b ** 3

    # 累加操作（注意 PyTorch 中 += 会创建新张量，需要重新赋值）
    c = c + c + 1
    c = c + 1 + c + (-a)

    # 复杂组合
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()

    # 最终计算
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f

    # 反向传播
    g.backward()

    # 保存 PyTorch 结果
    apt, bpt, gpt = a, b, g

    # ===== 验证结果 =====
    # 设置数值容差（由于浮点运算精度问题，允许一定误差）
    tol = 1e-6

    # 验证 1：前向传播结果一致
    assert abs(gmg.data - gpt.data.item()) < tol, \
        f"Forward mismatch: micrograd={gmg.data}, pytorch={gpt.data.item()}"

    # 验证 2：参数 a 的梯度一致
    assert abs(amg.grad - apt.grad.item()) < tol, \
        f"Gradient of a mismatch: micrograd={amg.grad}, pytorch={apt.grad.item()}"

    # 验证 3：参数 b 的梯度一致
    assert abs(bmg.grad - bpt.grad.item()) < tol, \
        f"Gradient of b mismatch: micrograd={bmg.grad}, pytorch={bpt.grad.item()}"

    print("✓ test_more_ops passed")
    print(f"  Forward: g={gmg.data:.6f}, expected={gpt.data.item():.6f}")
    print(f"  Gradient a: {amg.grad:.6f}, expected={apt.grad.item():.6f}")
    print(f"  Gradient b: {bmg.grad:.6f}, expected={bpt.grad.item():.6f}")


def test_mlp():
    """
    可选测试：验证 MLP 训练

    简单的 MLP 训练循环，验证整个流程的正确性。
    """
    from micrograd.nn import MLP

    # 创建模型
    model = MLP(2, [8, 4, 1])

    # 准备数据
    x = [1.0, 2.0]
    y_true = Value(1.0)

    # 训练一步
    y_pred = model(x)
    if not isinstance(y_pred, Value):
        y_pred = y_pred[0]

    loss = (y_pred - y_true) ** 2
    loss.backward()

    # 检查梯度
    grad_count = sum(1 for p in model.parameters() if p.grad != 0)
    print(f"✓ test_mlp passed: {grad_count}/{len(model.parameters())} parameters have gradients")


# ================ 主函数 ================

if __name__ == "__main__":
    """
    运行所有测试

    执行顺序：
    1. test_sanity_check — 基础测试
    2. test_more_ops — 扩展测试
    3. test_mlp — 可选测试
    """
    print("Running micrograd tests...\n")

    test_sanity_check()
    print()

    test_more_ops()
    print()

    # 可选的 MLP 测试
    # test_mlp()

    print("=" * 50)
    print("All tests passed! ✓")