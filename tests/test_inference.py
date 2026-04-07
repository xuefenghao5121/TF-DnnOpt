"""
端到端推理测试

验证 DNN-Opt 推理引擎的完整功能和正确性。
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tensorflow as tf
from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul, is_dnnopt_available, DnnoptConv2D
from inference_engine import DnnoptInferenceEngine, BatchInferenceEngine
from model_converter import ModelConverter, convert_keras_model


def test_simple_conv_model():
    """测试简单的 Conv 模型"""
    print("=" * 60)
    print("测试 1: 简单 Conv 模型")
    print("=" * 60)

    # 创建简单的 Conv 模型
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='simple_conv')

    # 生成测试数据
    np.random.seed(42)
    test_input = np.random.randn(4, 32, 32, 3).astype(np.float32)

    # 原始模型推理
    original_output = model(test_input).numpy()

    # 使用 DnnoptConv2D 替换
    dnnopt_inputs = tf.keras.Input(shape=(32, 32, 3))
    x = DnnoptConv2D(64, 3, padding='same', post_op='relu')(dnnopt_inputs)
    x = DnnoptConv2D(128, 3, padding='same', post_op='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    dnnopt_outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    dnnopt_model = tf.keras.Model(dnnopt_inputs, dnnopt_outputs, name='simple_conv_dnnopt')

    # 复制权重
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            dnnopt_model.layers[i].set_weights(layer.get_weights())

    # DNN-Opt 模型推理
    dnnopt_output = dnnopt_model(test_input).numpy()

    # 比较
    max_diff = np.max(np.abs(original_output - dnnopt_output))
    mean_diff = np.mean(np.abs(original_output - dnnopt_output))

    print(f"原始输出形状: {original_output.shape}")
    print(f"DNN-Opt 输出形状: {dnnopt_output.shape}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✓ 测试通过")
        return True
    else:
        print("✗ 测试失败")
        return False


def test_inference_engine():
    """测试推理引擎"""
    print("\n" + "=" * 60)
    print("测试 2: 推理引擎")
    print("=" * 60)

    # 创建简单模型
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # 保存模型
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model')
        model.save(model_path, save_format='tf')
        print(f"模型保存到: {model_path}")

        # 创建推理引擎
        engine = DnnoptInferenceEngine(model_path, verbose=True)

        # 测试推理
        np.random.seed(42)
        test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)

        result = engine.run(test_input)
        print(f"输出形状: {result.output.shape}")
        print(f"延迟: {result.latency_ms:.2f} ms")
        print(f"后端: {result.backend}")

        # 基准测试
        stats = engine.benchmark(test_input, iterations=50, warmup=5)
        print(f"平均延迟: {stats['mean_ms']:.2f} ms")
        print(f"吞吐量: {stats['throughput_fps']:.2f} FPS")

        print("✓ 推理引擎测试通过")
        return True


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "=" * 60)
    print("测试 3: 批量推理")
    print("=" * 60)

    # 创建模型
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs, outputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'batch_model')
        model.save(model_path, save_format='tf')

        # 创建批量推理引擎
        batch_engine = BatchInferenceEngine(model_path, batch_size=8)

        # 测试批量推理
        np.random.seed(42)
        test_inputs = np.random.randn(32, 32, 32, 3).astype(np.float32)

        outputs = batch_engine.infer(test_inputs)
        print(f"输入批次大小: {test_inputs.shape[0]}")
        print(f"输出形状: {outputs.shape}")

        if outputs.shape[0] == 32:
            print("✓ 批量推理测试通过")
            return True
        else:
            print("✗ 批量推理测试失败")
            return False


def test_model_converter():
    """测试模型转换器"""
    print("\n" + "=" * 60)
    print("测试 4: 模型转换器")
    print("=" * 60)

    # 创建模型
    inputs = tf.keras.Input(shape=(56, 56, 64))
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs, outputs, name='test_model')

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.path.join(tmpdir, 'original_model')
        converted_path = os.path.join(tmpdir, 'converted_model')

        # 保存原始模型
        model.save(original_path, save_format='tf')

        # 转换模型
        converter = ModelConverter(verbose=True)
        stats = converter.convert_savedmodel(original_path, converted_path)

        print(f"\n转换统计:")
        print(f"  总节点数: {stats.total_nodes}")
        print(f"  Conv2D 节点: {stats.conv2d_nodes}")
        print(f"  MatMul 节点: {stats.matmul_nodes}")
        print(f"  转换 Conv2D: {stats.converted_conv2d}")
        print(f"  转换 MatMul: {stats.converted_matmul}")
        print(f"  融合 ReLU: {stats.fused_relu}")

        # 验证转换后的模型可以加载
        if os.path.exists(os.path.join(converted_path, 'optimized_graph.pb')):
            print("✓ 转换后的 GraphDef 存在")
            return True
        else:
            print("✗ 转换失败")
            return False


def test_resnet_like_model():
    """测试 ResNet-like 模型"""
    print("\n" + "=" * 60)
    print("测试 5: ResNet-like 模型")
    print("=" * 60)

    # 创建简化版 ResNet-like 模型
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Conv1
    x = DnnoptConv2D(64, 7, strides=2, padding='same', post_op='relu', name='conv1')(inputs)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Conv2_x
    x = DnnoptConv2D(64, 3, padding='same', post_op='relu')(x)
    x = DnnoptConv2D(64, 3, padding='same')(x)

    # Global Pooling + FC
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='resnet_like_dnnopt')

    print(f"模型参数量: {model.count_params():,}")

    # 测试推理
    np.random.seed(42)
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

    # Warmup
    for _ in range(3):
        _ = model(test_input)

    # 推理
    import time
    start = time.perf_counter()
    output = model(test_input)
    end = time.perf_counter()

    print(f"输出形状: {output.shape}")
    print(f"推理时间: {(end - start) * 1000:.2f} ms")
    print(f"Top-5 预测: {np.argsort(output[0])[-5:][::-1]}")

    print("✓ ResNet-like 模型测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("DNN-Opt 端到端测试")
    print("=" * 60)
    print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    print()

    results = {}

    tests = [
        ('简单 Conv 模型', test_simple_conv_model),
        ('推理引擎', test_inference_engine),
        ('批量推理', test_batch_inference),
        ('模型转换器', test_model_converter),
        ('ResNet-like 模型', test_resnet_like_model),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"测试 '{name}' 失败: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    print()
    print(f"通过: {passed}/{total}")

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
