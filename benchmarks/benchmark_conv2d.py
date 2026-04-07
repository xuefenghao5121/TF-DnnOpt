"""
Conv2D 性能基准测试

对比 DNN-Opt Conv2D 与 TensorFlow 原生 Conv2D 的性能。
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dnnopt_ops import dnnopt_conv2d, is_dnnopt_available


@dataclass
class BenchConfig:
    """基准测试配置"""
    name: str
    input_shape: Tuple[int, int, int, int]  # [N, H, W, C]
    filter_shape: Tuple[int, int, int, int]  # [KH, KW, IC, OC]
    strides: Tuple[int, int, int, int]
    padding: str
    iterations: int = 100
    warmup: int = 10


# ResNet-like 配置
RESNET_CONFIGS = [
    BenchConfig(
        name="ResNet-Conv1",
        input_shape=(1, 224, 224, 3),
        filter_shape=(7, 7, 3, 64),
        strides=(1, 2, 2, 1),
        padding='SAME'
    ),
    BenchConfig(
        name="ResNet-3x3-64",
        input_shape=(1, 56, 56, 64),
        filter_shape=(3, 3, 64, 64),
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
    BenchConfig(
        name="ResNet-3x3-128",
        input_shape=(1, 28, 28, 128),
        filter_shape=(3, 3, 128, 128),
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
    BenchConfig(
        name="ResNet-3x3-256",
        input_shape=(1, 14, 14, 256),
        filter_shape=(3, 3, 256, 256),
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
    BenchConfig(
        name="ResNet-1x1-256",
        input_shape=(1, 56, 56, 256),
        filter_shape=(1, 1, 256, 64),
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
    BenchConfig(
        name="ResNet-1x1-64",
        input_shape=(1, 56, 56, 64),
        filter_shape=(1, 1, 64, 256),
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
]

# BERT-like 配置
BERT_CONFIGS = [
    BenchConfig(
        name="BERT-MatMul-768x768",
        input_shape=(1, 128, 768),
        filter_shape=(768, 768),  # 这实际上是 MatMul 形状
        strides=(1, 1, 1, 1),
        padding='SAME'
    ),
]


def benchmark_conv2d_tf(config: BenchConfig) -> Tuple[float, float]:
    """TensorFlow 原生 Conv2D 基准测试"""
    # 生成随机数据
    input_data = np.random.randn(*config.input_shape).astype(np.float32)
    filter_data = np.random.randn(*config.filter_shape).astype(np.float32)

    # 转换为 Tensor
    input_tensor = tf.constant(input_data)
    filter_tensor = tf.constant(filter_data)

    # 预热
    for _ in range(config.warmup):
        _ = tf.nn.conv2d(
            input_tensor, filter_tensor,
            strides=config.strides,
            padding=config.padding
        ).numpy()

    # 基准测试
    latencies = []
    for _ in range(config.iterations):
        start = time.perf_counter()
        _ = tf.nn.conv2d(
            input_tensor, filter_tensor,
            strides=config.strides,
            padding=config.padding
        ).numpy()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return np.mean(latencies), np.std(latencies)


def benchmark_conv2d_dnnopt(config: BenchConfig) -> Tuple[float, float]:
    """DNN-Opt Conv2D 基准测试"""
    # 生成随机数据
    input_data = np.random.randn(*config.input_shape).astype(np.float32)
    filter_data = np.random.randn(*config.filter_shape).astype(np.float32)

    # 预热
    for _ in range(config.warmup):
        _ = dnnopt_conv2d(
            input_data, filter_data,
            strides=config.strides,
            padding=config.padding
        ).numpy()

    # 基准测试
    latencies = []
    for _ in range(config.iterations):
        start = time.perf_counter()
        _ = dnnopt_conv2d(
            input_data, filter_data,
            strides=config.strides,
            padding=config.padding
        ).numpy()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return np.mean(latencies), np.std(latencies)


def run_benchmark(configs: List[BenchConfig], verbose: bool = True):
    """运行基准测试"""
    print("=" * 80)
    print("DNN-Opt Conv2D 性能基准测试")
    print("=" * 80)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    print()

    results = []

    for config in configs:
        if verbose:
            print(f"测试: {config.name}")
            print(f"  输入: {config.input_shape}, Filter: {config.filter_shape}")

        # TensorFlow 基准
        tf_mean, tf_std = benchmark_conv2d_tf(config)

        if verbose:
            print(f"  TensorFlow: {tf_mean:.2f} ms (±{tf_std:.2f})")

        # DNN-Opt 基准
        if is_dnnopt_available():
            dnnopt_mean, dnnopt_std = benchmark_conv2d_dnnopt(config)

            speedup = tf_mean / dnnopt_mean

            if verbose:
                print(f"  DNN-Opt:   {dnnopt_mean:.2f} ms (±{dnnopt_std:.2f})")
                print(f"  加速比:    {speedup:.2f}x")

            results.append({
                'name': config.name,
                'tf_mean': tf_mean,
                'tf_std': tf_std,
                'dnnopt_mean': dnnopt_mean,
                'dnnopt_std': dnnopt_std,
                'speedup': speedup
            })
        else:
            results.append({
                'name': config.name,
                'tf_mean': tf_mean,
                'tf_std': tf_std,
                'dnnopt_mean': None,
                'dnnopt_std': None,
                'speedup': None
            })

        if verbose:
            print()

    # 汇总
    print("=" * 80)
    print("汇总结果")
    print("=" * 80)

    if is_dnnopt_available():
        speedups = [r['speedup'] for r in results if r['speedup'] is not None]
        print(f"平均加速比: {np.mean(speedups):.2f}x")
        print(f"最大加速比: {np.max(speedups):.2f}x")
        print(f"最小加速比: {np.min(speedups):.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description='DNN-Opt Conv2D 性能基准测试')
    parser.add_argument('--config', type=str, default='resnet',
                        choices=['resnet', 'bert', 'all'],
                        help='测试配置')
    parser.add_argument('--iterations', type=int, default=100,
                        help='迭代次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数')
    args = parser.parse_args()

    # 选择配置
    if args.config == 'resnet':
        configs = RESNET_CONFIGS
    elif args.config == 'bert':
        configs = BERT_CONFIGS
    else:
        configs = RESNET_CONFIGS + BERT_CONFIGS

    # 更新迭代次数
    for config in configs:
        config.iterations = args.iterations
        config.warmup = args.warmup

    run_benchmark(configs)


if __name__ == '__main__':
    main()
