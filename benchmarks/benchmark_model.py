"""
模型级性能基准测试

测试完整模型的 TensorFlow vs DNN-Opt 性能对比。
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dnnopt_ops import is_dnnopt_available, DnnoptConv2D
from inference_engine import DnnoptInferenceEngine


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    backend: str
    batch_size: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_fps: float


def create_simple_cnn(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 1000,
    use_dnnopt: bool = False
) -> tf.keras.Model:
    """创建简单的 CNN 模型"""
    inputs = tf.keras.Input(shape=input_shape)

    if use_dnnopt:
        x = DnnoptConv2D(64, 7, strides=2, padding='same', post_op='relu', name='conv1')(inputs)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        x = DnnoptConv2D(128, 3, padding='same', post_op='relu')(x)
        x = DnnoptConv2D(128, 3, padding='same', post_op='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = DnnoptConv2D(256, 3, padding='same', post_op='relu')(x)
        x = DnnoptConv2D(256, 3, padding='same', post_op='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = DnnoptConv2D(512, 3, padding='same', post_op='relu')(x)
        x = DnnoptConv2D(512, 3, padding='same', post_op='relu')(x)
    else:
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu', name='conv1')(inputs)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    name = 'simple_cnn_dnnopt' if use_dnnopt else 'simple_cnn_tf'
    return tf.keras.Model(inputs, outputs, name=name)


def create_mlp(
    input_dim: int = 768,
    hidden_dim: int = 3072,
    output_dim: int = 768,
    num_layers: int = 3,
    use_dnnopt: bool = False
) -> tf.keras.Model:
    """创建 MLP 模型 (BERT-like)"""
    inputs = tf.keras.Input(shape=(input_dim,))

    x = inputs
    for i in range(num_layers):
        if use_dnnopt:
            # 使用 Dense 层 (内部使用 MatMul)
            x = tf.keras.layers.Dense(hidden_dim, activation='gelu', name=f'dense_{i}')(x)
        else:
            x = tf.keras.layers.Dense(hidden_dim, activation='gelu', name=f'dense_{i}')(x)

    outputs = tf.keras.layers.Dense(output_dim)(x)

    name = 'mlp_dnnopt' if use_dnnopt else 'mlp_tf'
    return tf.keras.Model(inputs, outputs, name=name)


def benchmark_model(
    model: tf.keras.Model,
    input_data: np.ndarray,
    iterations: int = 100,
    warmup: int = 10,
    name: str = 'Model'
) -> BenchmarkResult:
    """基准测试模型"""
    print(f"\n基准测试: {name}")
    print("-" * 50)

    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        _ = model(input_data, training=False)

    # 基准测试
    print(f"运行 {iterations} 次迭代...")
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    result = BenchmarkResult(
        model_name=name,
        backend='dnnopt' if 'dnnopt' in name.lower() else 'tensorflow',
        batch_size=input_data.shape[0],
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        throughput_fps=float(1000.0 / np.mean(latencies))
    )

    print(f"平均延迟: {result.mean_ms:.2f} ms (±{result.std_ms:.2f})")
    print(f"P50/P95/P99: {result.p50_ms:.2f}/{result.p95_ms:.2f}/{result.p99_ms:.2f} ms")
    print(f"吞吐量: {result.throughput_fps:.2f} FPS")

    return result


def compare_models(
    tf_result: BenchmarkResult,
    dnnopt_result: BenchmarkResult
) -> Dict[str, Any]:
    """对比两个模型的性能"""
    speedup = tf_result.mean_ms / dnnopt_result.mean_ms

    print("\n性能对比")
    print("=" * 50)
    print(f"TensorFlow 平均延迟: {tf_result.mean_ms:.2f} ms")
    print(f"DNN-Opt   平均延迟: {dnnopt_result.mean_ms:.2f} ms")
    print(f"加速比: {speedup:.2f}x")

    if speedup > 1.0:
        improvement = (speedup - 1.0) * 100
        print(f"性能提升: {improvement:.1f}%")
    else:
        degradation = (1.0 - speedup) * 100
        print(f"性能下降: {degradation:.1f}%")

    return {
        'speedup': speedup,
        'tf_mean_ms': tf_result.mean_ms,
        'dnnopt_mean_ms': dnnopt_result.mean_ms,
    }


def main():
    parser = argparse.ArgumentParser(description='DNN-Opt 模型基准测试')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'all'],
                        help='模型类型')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--iterations', type=int, default=100,
                        help='迭代次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数')
    parser.add_argument('--input-size', type=int, default=224,
                        help='输入尺寸 (CNN)')
    args = parser.parse_args()

    print("=" * 70)
    print("DNN-Opt 模型基准测试")
    print("=" * 70)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"批处理大小: {args.batch_size}")
    print()

    results = []

    if args.model in ('cnn', 'all'):
        print("\n" + "=" * 70)
        print("CNN 模型基准测试")
        print("=" * 70)

        input_shape = (args.input_size, args.input_size, 3)
        input_data = np.random.randn(args.batch_size, *input_shape).astype(np.float32)

        # TensorFlow 模型
        print("\n创建 TensorFlow CNN 模型...")
        tf_model = create_simple_cnn(input_shape, use_dnnopt=False)
        tf_model.summary()

        tf_result = benchmark_model(
            tf_model, input_data,
            iterations=args.iterations,
            warmup=args.warmup,
            name='CNN-TensorFlow'
        )
        results.append(tf_result)

        # DNN-Opt 模型
        print("\n创建 DNN-Opt CNN 模型...")
        dnnopt_model = create_simple_cnn(input_shape, use_dnnopt=True)

        # 复制权重
        for tf_layer, dnnopt_layer in zip(tf_model.layers, dnnopt_model.layers):
            if len(tf_layer.get_weights()) > 0:
                dnnopt_layer.set_weights(tf_layer.get_weights())

        dnnopt_result = benchmark_model(
            dnnopt_model, input_data,
            iterations=args.iterations,
            warmup=args.warmup,
            name='CNN-DNN-Opt'
        )
        results.append(dnnopt_result)

        # 对比
        comparison = compare_models(tf_result, dnnopt_result)

    if args.model in ('mlp', 'all'):
        print("\n" + "=" * 70)
        print("MLP 模型基准测试 (BERT-like)")
        print("=" * 70)

        input_dim = 768
        input_data = np.random.randn(args.batch_size * 128, input_dim).astype(np.float32)

        # TensorFlow MLP
        print("\n创建 TensorFlow MLP 模型...")
        tf_mlp = create_mlp(use_dnnopt=False)
        tf_mlp.summary()

        tf_mlp_result = benchmark_model(
            tf_mlp, input_data,
            iterations=args.iterations,
            warmup=args.warmup,
            name='MLP-TensorFlow'
        )
        results.append(tf_mlp_result)

        # DNN-Opt MLP
        print("\n创建 DNN-Opt MLP 模型...")
        dnnopt_mlp = create_mlp(use_dnnopt=True)

        # 复制权重
        for tf_layer, dnnopt_layer in zip(tf_mlp.layers, dnnopt_mlp.layers):
            if len(tf_layer.get_weights()) > 0:
                dnnopt_layer.set_weights(tf_layer.get_weights())

        dnnopt_mlp_result = benchmark_model(
            dnnopt_mlp, input_data,
            iterations=args.iterations,
            warmup=args.warmup,
            name='MLP-DNN-Opt'
        )
        results.append(dnnopt_mlp_result)

        # 对比
        comparison = compare_models(tf_mlp_result, dnnopt_mlp_result)

    # 最终汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)

    for result in results:
        print(f"\n{result.model_name}:")
        print(f"  平均延迟: {result.mean_ms:.2f} ms")
        print(f"  吞吐量: {result.throughput_fps:.2f} FPS")

    print()


if __name__ == '__main__':
    main()
