"""
Conv2D 正确性测试

验证 DNN-Opt Conv2D 与 TensorFlow 原生 Conv2D 结果一致性。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tensorflow as tf
from dnnopt_ops import dnnopt_conv2d, is_dnnopt_available

# 测试配置
TEST_CONFIGS = [
    # (input_shape, filter_shape, strides, padding, post_op)
    # 基本测试
    ((1, 32, 32, 3), (3, 3, 3, 64), (1, 1, 1, 1), 'SAME', 'none'),
    ((1, 32, 32, 3), (3, 3, 3, 64), (1, 2, 2, 1), 'SAME', 'none'),
    ((1, 32, 32, 3), (3, 3, 3, 64), (1, 1, 1, 1), 'VALID', 'none'),

    # 大通道测试
    ((1, 56, 56, 64), (3, 3, 64, 128), (1, 1, 1, 1), 'SAME', 'none'),
    ((1, 56, 56, 64), (3, 3, 64, 128), (1, 2, 2, 1), 'SAME', 'none'),

    # 1x1 卷积
    ((1, 56, 56, 256), (1, 1, 256, 64), (1, 1, 1, 1), 'SAME', 'none'),

    # Post-op 测试
    ((1, 32, 32, 3), (3, 3, 3, 64), (1, 1, 1, 1), 'SAME', 'relu'),
    ((1, 32, 32, 3), (3, 3, 3, 64), (1, 1, 1, 1), 'SAME', 'relu6'),

    # Batch 测试
    ((4, 32, 32, 3), (3, 3, 3, 64), (1, 1, 1, 1), 'SAME', 'none'),

    # 大 kernel 测试
    ((1, 224, 224, 3), (7, 7, 3, 64), (1, 2, 2, 1), 'SAME', 'none'),
]


def test_conv2d_correctness():
    """测试 Conv2D 正确性"""
    print("=" * 60)
    print("DNN-Opt Conv2D 正确性测试")
    print("=" * 60)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    print()

    all_passed = True

    for i, (input_shape, filter_shape, strides, padding, post_op) in enumerate(TEST_CONFIGS):
        print(f"测试 {i+1}/{len(TEST_CONFIGS)}: input={input_shape}, filter={filter_shape}, "
              f"strides={strides}, padding={padding}, post_op={post_op}")

        try:
            # 生成随机数据
            np.random.seed(42 + i)
            input_data = np.random.randn(*input_shape).astype(np.float32)
            filter_data = np.random.randn(*filter_shape).astype(np.float32)

            # 计算输出通道数
            oc = filter_shape[3]
            bias_data = np.random.randn(oc).astype(np.float32)

            # TensorFlow 原生 Conv2D
            tf_out = tf.nn.conv2d(
                input_data, filter_data,
                strides=strides,
                padding=padding,
                data_format='NHWC'
            )
            tf_out = tf.nn.bias_add(tf_out, bias_data)

            if post_op == 'relu':
                tf_out = tf.nn.relu(tf_out)
            elif post_op == 'relu6':
                tf_out = tf.nn.relu6(tf_out)

            # DNN-Opt Conv2D
            dnnopt_out = dnnopt_conv2d(
                input_data, filter_data, bias_data,
                strides=strides,
                padding=padding,
                post_op=post_op
            )

            # 比较结果
            tf_out_np = tf_out.numpy()
            dnnopt_out_np = dnnopt_out.numpy()

            # 检查形状
            if tf_out_np.shape != dnnopt_out_np.shape:
                print(f"  ✗ 形状不匹配: TF={tf_out_np.shape}, DNN-Opt={dnnopt_out_np.shape}")
                all_passed = False
                continue

            # 检查数值
            max_diff = np.max(np.abs(tf_out_np - dnnopt_out_np))
            mean_diff = np.mean(np.abs(tf_out_np - dnnopt_out_np))

            if max_diff < 1e-4:  # 放宽容差
                print(f"  ✓ 通过 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
            else:
                print(f"  ✗ 失败 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                all_passed = False

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("存在失败的测试")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    passed = test_conv2d_correctness()
    sys.exit(0 if passed else 1)
