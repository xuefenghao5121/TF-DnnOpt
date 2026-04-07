"""
MatMul 正确性测试

验证 DNN-Opt MatMul 与 TensorFlow 原生 MatMul 结果一致性。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tensorflow as tf
from dnnopt_ops import dnnopt_matmul, is_dnnopt_available

# 测试配置
# (shape_a, shape_b, transpose_a, transpose_b)
TEST_CONFIGS = [
    # 基本矩阵乘法
    ((128, 256), (256, 512), False, False),
    ((64, 128), (128, 256), False, False),
    ((32, 64), (64, 128), False, False),

    # 方阵
    ((256, 256), (256, 256), False, False),
    ((512, 512), (512, 512), False, False),

    # 非方阵
    ((128, 64), (64, 256), False, False),
    ((256, 128), (128, 512), False, False),

    # 转置测试
    ((256, 128), (256, 64), False, True),   # B^T
    ((128, 256), (64, 256), True, False),   # A^T
    ((256, 128), (128, 256), True, True),   # A^T * B^T

    # 小矩阵
    ((16, 32), (32, 64), False, False),
    ((8, 16), (16, 32), False, False),

    # 大矩阵
    ((1024, 512), (512, 256), False, False),
    ((768, 768), (768, 768), False, False),  # BERT-like

    # 极端比例
    ((1024, 64), (64, 1024), False, False),
    ((64, 1024), (1024, 64), False, False),
]


def test_matmul_correctness():
    """测试 MatMul 正确性"""
    print("=" * 60)
    print("DNN-Opt MatMul 正确性测试")
    print("=" * 60)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    print()

    all_passed = True
    failed_tests = []

    for i, (shape_a, shape_b, transpose_a, transpose_b) in enumerate(TEST_CONFIGS):
        print(f"测试 {i+1}/{len(TEST_CONFIGS)}: A={shape_a}, B={shape_b}, "
              f"transpose_a={transpose_a}, transpose_b={transpose_b}")

        try:
            # 生成随机数据
            np.random.seed(42 + i)
            a_data = np.random.randn(*shape_a).astype(np.float32)
            b_data = np.random.randn(*shape_b).astype(np.float32)

            # TensorFlow 原生 MatMul
            tf_out = tf.matmul(
                a_data, b_data,
                transpose_a=transpose_a,
                transpose_b=transpose_b
            )

            # DNN-Opt MatMul
            dnnopt_out = dnnopt_matmul(
                a_data, b_data,
                transpose_a=transpose_a,
                transpose_b=transpose_b
            )

            # 比较结果
            tf_out_np = tf_out.numpy()
            dnnopt_out_np = dnnopt_out.numpy()

            # 检查形状
            expected_shape = tf_out_np.shape
            if dnnopt_out_np.shape != expected_shape:
                print(f"  ✗ 形状不匹配: 期望={expected_shape}, 实际={dnnopt_out_np.shape}")
                all_passed = False
                failed_tests.append(i)
                continue

            # 检查数值
            max_diff = np.max(np.abs(tf_out_np - dnnopt_out_np))
            mean_diff = np.mean(np.abs(tf_out_np - dnnopt_out_np))

            # 相对误差
            rel_diff = max_diff / (np.max(np.abs(tf_out_np)) + 1e-8)

            if max_diff < 1e-4 and rel_diff < 1e-4:
                print(f"  ✓ 通过 (max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e})")
            else:
                print(f"  ✗ 失败 (max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e})")
                all_passed = False
                failed_tests.append(i)

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            all_passed = False
            failed_tests.append(i)

    print()
    print("=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print(f"存在失败的测试: {failed_tests}")
    print("=" * 60)

    return all_passed


def test_matmul_performance():
    """测试 MatMul 性能 (可选)"""
    print("\n" + "=" * 60)
    print("DNN-Opt MatMul 性能测试")
    print("=" * 60)

    if not is_dnnopt_available():
        print("DNN-Opt 不可用，跳过性能测试")
        return

    # BERT-like GEMM
    shape_a = (128, 768)  # Batch * SeqLen, HiddenDim
    shape_b = (768, 768)  # HiddenDim, HiddenDim

    print(f"矩阵形状: A={shape_a}, B={shape_b}")

    np.random.seed(42)
    a_data = np.random.randn(*shape_a).astype(np.float32)
    b_data = np.random.randn(*shape_b).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = tf.matmul(a_data, b_data)
        _ = dnnopt_matmul(a_data, b_data)

    # TensorFlow benchmark
    iterations = 100
    tf_times = []
    for _ in range(iterations):
        start = tf.timestamp()
        _ = tf.matmul(a_data, b_data)
        end = tf.timestamp()
        tf_times.append((end - start).numpy() * 1000)

    # DNN-Opt benchmark
    dnnopt_times = []
    for _ in range(iterations):
        start = tf.timestamp()
        _ = dnnopt_matmul(a_data, b_data)
        end = tf.timestamp()
        dnnopt_times.append((end - start).numpy() * 1000)

    tf_mean = np.mean(tf_times)
    dnnopt_mean = np.mean(dnnopt_times)

    print(f"TensorFlow 平均延迟: {tf_mean:.3f} ms")
    print(f"DNN-Opt   平均延迟: {dnnopt_mean:.3f} ms")

    if dnnopt_mean > 0:
        speedup = tf_mean / dnnopt_mean
        print(f"加速比: {speedup:.2f}x")


if __name__ == '__main__':
    passed = test_matmul_correctness()

    # 可选性能测试
    if '--perf' in sys.argv:
        test_matmul_performance()

    sys.exit(0 if passed else 1)
