"""
DNN-Opt TensorFlow 无感知集成示例

用户只需在代码开头导入 dnnopt_tensorflow，之后的代码完全不变。
"""

# 方法 1: 在导入 tensorflow 之前导入
# import dnnopt_tensorflow

import sys
import os
import time

# 添加 python 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import tensorflow as tf
import numpy as np

# 方法 2: 在导入 tensorflow 之后导入
import dnnopt_tensorflow


def demo_conv2d():
    """演示 Conv2D 无感知使用"""
    print("\n" + "=" * 60)
    print("Demo 1: tf.nn.conv2d")
    print("=" * 60)

    # 用户代码完全不变
    input_data = np.random.randn(1, 32, 32, 3).astype(np.float32)
    filter_data = np.random.randn(3, 3, 3, 64).astype(np.float32)

    # 正常使用 tf.nn.conv2d
    output = tf.nn.conv2d(
        input_data, filter_data,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )

    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"DNN-Opt 已启用: {dnnopt_tensorflow.is_patched()}")


def demo_keras_conv2d():
    """演示 Keras Conv2D 无感知使用"""
    print("\n" + "=" * 60)
    print("Demo 2: tf.keras.layers.Conv2D")
    print("=" * 60)

    # 用户代码完全不变
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()

    # 推理
    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
    output = model(input_data)

    print(f"输出形状: {output.shape}")


def demo_matmul():
    """演示 MatMul 无感知使用"""
    print("\n" + "=" * 60)
    print("Demo 3: tf.matmul")
    print("=" * 60)

    # 用户代码完全不变
    a = np.random.randn(128, 256).astype(np.float32)
    b = np.random.randn(256, 512).astype(np.float32)

    output = tf.matmul(a, b)

    print(f"A 形状: {a.shape}")
    print(f"B 形状: {b.shape}")
    print(f"输出形状: {output.shape}")


def demo_resnet50():
    """演示 ResNet50 无感知使用"""
    print("\n" + "=" * 60)
    print("Demo 4: ResNet50")
    print("=" * 60)

    # 用户代码完全不变
    model = tf.keras.applications.ResNet50(weights=None)

    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)

    # Warmup
    for _ in range(3):
        _ = model(input_data)

    # 推理
    start = time.perf_counter()
    output = model(input_data)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"输出形状: {output.shape}")
    print(f"推理时间: {elapsed:.2f} ms")


def demo_enable_disable():
    """演示启用/禁用功能"""
    print("\n" + "=" * 60)
    print("Demo 5: 启用/禁用 DNN-Opt")
    print("=" * 60)

    input_data = np.random.randn(1, 32, 32, 3).astype(np.float32)
    filter_data = np.random.randn(3, 3, 3, 64).astype(np.float32)

    # 当前状态
    print(f"DNN-Opt 已启用: {dnnopt_tensorflow.is_patched()}")

    # 正常推理 (使用 DNN-Opt)
    output1 = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='SAME')

    # 禁用 DNN-Opt
    dnnopt_tensorflow.disable()
    print(f"DNN-Opt 已启用: {dnnopt_tensorflow.is_patched()}")

    # 使用原生 TensorFlow
    output2 = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='SAME')

    # 重新启用
    dnnopt_tensorflow.enable()
    print(f"DNN-Opt 已启用: {dnnopt_tensorflow.is_patched()}")

    # 比较结果 (应该相同)
    diff = np.max(np.abs(output1.numpy() - output2.numpy()))
    print(f"结果差异: {diff:.6f}")


def main():
    print("=" * 60)
    print("DNN-Opt TensorFlow 无感知集成演示")
    print("=" * 60)
    print(f"DNN-Opt 可用: {dnnopt_tensorflow.is_dnnopt_available()}")
    print(f"DNN-Opt 已启用: {dnnopt_tensorflow.is_patched()}")

    demo_conv2d()
    demo_keras_conv2d()
    demo_matmul()

    # 可选: ResNet50
    try:
        demo_resnet50()
    except Exception as e:
        print(f"ResNet50 demo 跳过: {e}")

    demo_enable_disable()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("""
使用说明:
---------
1. 在代码开头添加: import dnnopt_tensorflow
2. 之后的 TensorFlow 代码完全不变
3. 所有 tf.nn.conv2d, tf.matmul, tf.keras.layers.Conv2D 自动使用 DNN-Opt

环境变量:
---------
DNNOPT_DISABLE=1   # 禁用 DNN-Opt
DNNOPT_VERBOSE=1   # 显示详细日志

API:
----
dnnopt_tensorflow.enable()   # 启用 DNN-Opt
dnnopt_tensorflow.disable()  # 禁用，恢复原生 TensorFlow
""")


if __name__ == '__main__':
    main()
