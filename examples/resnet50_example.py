"""
ResNet50 推理示例

展示如何使用 DNN-Opt 推理引擎进行图像分类推理。
"""

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf

# 添加 python 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from inference_engine import DnnoptInferenceEngine
from dnnopt_ops import is_dnnopt_available, DnnoptConv2D


def create_resnet50_model():
    """创建 ResNet50 模型"""
    print("创建 ResNet50 模型...")

    # 使用 Keras 预训练模型
    model = tf.keras.applications.ResNet50(weights='imagenet')

    return model


def build_dnnopt_resnet50():
    """使用 DNN-Opt Conv2D 构建 ResNet50"""
    print("构建 DNN-Opt 优化版 ResNet50...")

    inputs = tf.keras.Input(shape=(224, 224, 3))

    # 预处理
    x = tf.keras.layers.Lambda(
        lambda img: tf.keras.applications.resnet50.preprocess_input(img)
    )(inputs)

    # 使用 DNN-Opt Conv2D 替换标准 Conv2D
    # 注意: 这是一个简化示例，实际 ResNet50 有很多层

    # Conv1: 7x7, 64, stride 2
    x = DnnoptConv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='SAME',
        post_op='relu',
        name='conv1_conv'
    )(x)

    # MaxPool
    x = tf.keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='SAME'
    )(x)

    # ... 这里应该继续添加 ResNet50 的其余层
    # 为简化示例，我们直接使用 Dense 层作为输出

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # FC
    outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='resnet50_dnnopt')

    return model


def prepare_dummy_input(batch_size=1):
    """准备测试输入数据"""
    # 生成随机图像数据
    images = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
    return images


def benchmark_model(model, inputs, iterations=100, warmup=10, name='Model'):
    """基准测试模型"""
    print(f"\n基准测试: {name}")
    print("=" * 50)

    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        _ = model(inputs)

    # 基准测试
    print(f"运行 {iterations} 次迭代...")
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = model(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    print(f"结果:")
    print(f"  平均延迟: {np.mean(latencies):.2f} ms")
    print(f"  标准差:   {np.std(latencies):.2f} ms")
    print(f"  最小:     {np.min(latencies):.2f} ms")
    print(f"  最大:     {np.max(latencies):.2f} ms")
    print(f"  P95:      {np.percentile(latencies, 95):.2f} ms")
    print(f"  吞吐量:   {1000/np.mean(latencies):.2f} FPS")

    return latencies


def main():
    parser = argparse.ArgumentParser(description='ResNet50 DNN-Opt 示例')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'benchmark', 'convert'],
                        help='运行模式')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--iterations', type=int, default=100,
                        help='基准测试迭代次数')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径 (用于加载已保存的模型)')
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet50 + DNN-Opt 示例")
    print("=" * 60)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")

    if args.mode == 'demo':
        # 演示模式
        print("\n创建模型...")

        # 标准 ResNet50
        standard_model = create_resnet50_model()

        # 准备输入
        inputs = prepare_dummy_input(args.batch_size)
        print(f"输入形状: {inputs.shape}")

        # 运行推理
        print("\n运行推理...")
        output = standard_model(inputs)
        print(f"输出形状: {output.shape}")
        print(f"Top-5 预测类别: {np.argsort(output[0])[-5:][::-1]}")

        # 使用推理引擎
        print("\n使用 DnnoptInferenceEngine...")

        # 保存模型
        save_path = '/tmp/resnet50_demo'
        print(f"保存模型到: {save_path}")
        standard_model.save(save_path, save_format='tf')

        # 加载并推理
        engine = DnnoptInferenceEngine(save_path)
        result = engine.run(inputs)
        print(f"推理结果形状: {result.output.shape}")
        print(f"延迟: {result.latency_ms:.2f} ms")

    elif args.mode == 'benchmark':
        # 基准测试模式
        print("\n创建模型...")

        if args.model_path and os.path.exists(args.model_path):
            # 加载已保存的模型
            print(f"加载模型: {args.model_path}")
            engine = DnnoptInferenceEngine(args.model_path)
            inputs = prepare_dummy_input(args.batch_size)

            stats = engine.benchmark(inputs, iterations=args.iterations)

            print(f"\n基准测试结果:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            # 创建新模型
            standard_model = create_resnet50_model()
            inputs = prepare_dummy_input(args.batch_size)

            latencies = benchmark_model(
                standard_model, inputs,
                iterations=args.iterations,
                name='ResNet50 (TensorFlow)'
            )

    elif args.mode == 'convert':
        # 模型转换模式
        from model_converter import convert_savedmodel

        print("\n转换模型...")

        # 创建并保存原始模型
        model = create_resnet50_model()
        original_path = '/tmp/resnet50_original'
        model.save(original_path, save_format='tf')
        print(f"原始模型保存到: {original_path}")

        # 转换
        optimized_path = '/tmp/resnet50_dnnopt'
        stats = convert_savedmodel(original_path, optimized_path)

        print(f"\n转换完成!")
        print(f"  Conv2D 转换: {stats.converted_conv2d}/{stats.conv2d_nodes}")
        print(f"  MatMul 转换: {stats.converted_matmul}/{stats.matmul_nodes}")
        print(f"  优化模型保存到: {optimized_path}")


if __name__ == '__main__':
    main()
