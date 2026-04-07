#!/usr/bin/env python3
"""
DNN-Opt 测试运行脚本

运行所有测试并生成报告。
"""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def run_test(test_file: str, extra_args: List[str] = None) -> Tuple[bool, float, str]:
    """
    运行单个测试文件

    Returns:
        (success, duration, output)
    """
    cmd = [sys.executable, test_file]
    if extra_args:
        cmd.extend(extra_args)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )
        duration = time.perf_counter() - start
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, duration, output
    except subprocess.TimeoutExpired:
        return False, 300.0, "测试超时"
    except Exception as e:
        return False, 0.0, str(e)


def main():
    parser = argparse.ArgumentParser(description='运行 DNN-Opt 测试')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'conv2d', 'matmul', 'inference'],
                        help='要运行的测试')
    parser.add_argument('--perf', action='store_true',
                        help='包含性能测试')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    args = parser.parse_args()

    # 测试目录
    test_dir = os.path.dirname(__file__)

    # 测试文件映射
    test_files = {
        'conv2d': os.path.join(test_dir, 'test_conv2d_correctness.py'),
        'matmul': os.path.join(test_dir, 'test_matmul_correctness.py'),
        'inference': os.path.join(test_dir, 'test_inference.py'),
    }

    # 选择要运行的测试
    if args.test == 'all':
        tests_to_run = list(test_files.items())
    else:
        tests_to_run = [(args.test, test_files[args.test])]

    print("=" * 70)
    print("DNN-Opt 测试套件")
    print("=" * 70)
    print(f"Python: {sys.executable}")
    print(f"测试目录: {test_dir}")
    print()

    # 检查 TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow 版本: {tf.__version__}")
        print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("警告: TensorFlow 未安装")
        return 1

    # 检查 DNN-Opt
    try:
        from dnnopt_ops import is_dnnopt_available
        print(f"DNN-Opt 可用: {is_dnnopt_available()}")
    except ImportError as e:
        print(f"警告: 无法导入 dnnopt_ops: {e}")

    print()
    print("=" * 70)
    print()

    # 运行测试
    results = {}
    extra_args = ['--perf'] if args.perf else []

    for name, test_file in tests_to_run:
        if not os.path.exists(test_file):
            print(f"跳过 {name}: 测试文件不存在")
            continue

        print(f"运行 {name} 测试...")
        print("-" * 70)

        success, duration, output = run_test(test_file, extra_args)
        results[name] = {
            'success': success,
            'duration': duration,
            'output': output
        }

        if args.verbose:
            print(output)
        else:
            # 只打印最后几行
            lines = output.strip().split('\n')
            for line in lines[-10:]:
                print(line)

        status = "✓ 通过" if success else "✗ 失败"
        print(f"\n{status} ({duration:.2f}s)")
        print()

    # 汇总
    print("=" * 70)
    print("测试汇总")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    total_time = sum(r['duration'] for r in results.values())

    for name, result in results.items():
        status = "✓ 通过" if result['success'] else "✗ 失败"
        print(f"  {name}: {status} ({result['duration']:.2f}s)")

    print()
    print(f"总计: {passed}/{total} 通过, 耗时 {total_time:.2f}s")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
