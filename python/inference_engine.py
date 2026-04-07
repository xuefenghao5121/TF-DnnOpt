"""
DNN-Opt 推理引擎

提供高性能的模型推理封装，自动使用 DNN-Opt 优化的算子。

使用方法:
    from inference_engine import DnnoptInferenceEngine

    engine = DnnoptInferenceEngine('path/to/model')
    output = engine.run(input_data)
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json

import numpy as np
import tensorflow as tf

# 导入 DNN-Opt ops
try:
    from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul, is_dnnopt_available
except ImportError:
    # 相对导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul, is_dnnopt_available


@dataclass
class InferenceResult:
    """推理结果"""
    output: Any
    latency_ms: float
    backend: str  # 'dnnopt' or 'tensorflow'


@dataclass
class ProfileResult:
    """性能分析结果"""
    total_time_ms: float
    layer_times: Dict[str, float]
    ops_count: Dict[str, int]
    dnnopt_speedup: Optional[float] = None


class DnnoptInferenceEngine:
    """
    DNN-Opt 推理引擎

    支持两种模式:
    1. 加载已转换的模型 (包含 DnnoptConv2D/DnnoptMatMul 算子)
    2. 加载原始模型，运行时替换算子 (实验性)
    """

    def __init__(
        self,
        model_path: str,
        signature_key: str = 'serving_default',
        warmup_iterations: int = 5,
        verbose: bool = True
    ):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径 (SavedModel 目录或 GraphDef 文件)
            signature_key: 签名键名
            warmup_iterations: 预热迭代次数
            verbose: 是否输出详细信息
        """
        self.model_path = model_path
        self.signature_key = signature_key
        self.warmup_iterations = warmup_iterations
        self.verbose = verbose

        self.model = None
        self.concrete_func = None
        self.input_names = []
        self.output_names = []
        self._backend = 'dnnopt' if is_dnnopt_available() else 'tensorflow'

        self._load_model()
        self._warmup()

    def log(self, msg: str):
        """输出日志"""
        if self.verbose:
            print(f"[InferenceEngine] {msg}")

    def _load_model(self):
        """加载模型"""
        self.log(f"加载模型: {self.model_path}")
        self.log(f"后端: {self._backend}")

        if os.path.isdir(self.model_path):
            # SavedModel 目录
            self.model = tf.saved_model.load(self.model_path)

            if self.signature_key in self.model.signatures:
                self.concrete_func = self.model.signatures[self.signature_key]
            else:
                sig_keys = list(self.model.signatures.keys())
                if sig_keys:
                    self.signature_key = sig_keys[0]
                    self.concrete_func = self.model.signatures[self.signature_key]
                else:
                    raise ValueError(f"未找到签名 '{self.signature_key}'")

            # 获取输入输出名称
            self.input_names = list(self.concrete_func.structured_input_signature[1].keys())
            self.output_names = list(self.concrete_func.structured_outputs.keys())

        elif self.model_path.endswith('.pb'):
            # GraphDef 文件
            graph_def = tf.compat.v1.GraphDef()
            with open(self.model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())

            # 导入 GraphDef
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')

            self.log("警告: GraphDef 加载方式有限制，建议使用 SavedModel")

        else:
            raise ValueError(f"不支持的模型格式: {self.model_path}")

        self.log(f"输入: {self.input_names}")
        self.log(f"输出: {self.output_names}")

    def _warmup(self):
        """预热模型"""
        if self.warmup_iterations <= 0:
            return

        self.log(f"预热模型 ({self.warmup_iterations} 次)...")

        # 获取输入形状
        input_shapes = self._get_input_shapes()
        dummy_inputs = {}

        for name, shape in input_shapes.items():
            # 生成小尺寸的 dummy 输入
            dummy_shape = [1 if s is None or s <= 0 else min(s, 32) for s in shape]
            dummy_inputs[name] = np.random.randn(*dummy_shape).astype(np.float32)

        for i in range(self.warmup_iterations):
            _ = self.run(dummy_inputs, profile=False)

        self.log("预热完成")

    def _get_input_shapes(self) -> Dict[str, Tuple]:
        """获取输入形状"""
        shapes = {}
        for name in self.input_names:
            # 从 concrete function 获取形状信息
            spec = self.concrete_func.structured_input_signature[1].get(name)
            if spec is not None:
                shapes[name] = tuple(spec.shape.as_list())
            else:
                shapes[name] = (None, None, None, None)  # 默认 NHWC
        return shapes

    def run(
        self,
        inputs: Union[Dict[str, np.ndarray], np.ndarray, tf.Tensor],
        profile: bool = False
    ) -> Union[InferenceResult, Tuple[InferenceResult, ProfileResult]]:
        """
        运行推理

        Args:
            inputs: 输入数据，可以是字典、数组或 Tensor
            profile: 是否进行性能分析

        Returns:
            推理结果，如果 profile=True 则同时返回性能分析结果
        """
        # 准备输入
        if isinstance(inputs, np.ndarray) or isinstance(inputs, tf.Tensor):
            if len(self.input_names) == 1:
                inputs = {self.input_names[0]: inputs}
            else:
                raise ValueError(f"模型需要 {len(self.input_names)} 个输入，请提供字典")

        # 转换为 Tensor
        tensor_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                tensor_inputs[name] = tf.constant(value)
            else:
                tensor_inputs[name] = value

        # 运行推理
        start_time = time.perf_counter()

        if profile:
            # 性能分析模式
            with tf.profiler.experimental.Trace('inference'):
                outputs = self.concrete_func(**tensor_inputs)
        else:
            outputs = self.concrete_func(**tensor_inputs)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # 处理输出
        if isinstance(outputs, dict):
            output = {k: v.numpy() for k, v in outputs.items()}
        else:
            output = outputs.numpy() if hasattr(outputs, 'numpy') else outputs

        result = InferenceResult(
            output=output,
            latency_ms=latency_ms,
            backend=self._backend
        )

        if profile:
            profile_result = self._analyze_performance(outputs, latency_ms)
            return result, profile_result

        return result

    def _analyze_performance(
        self,
        outputs: Any,
        total_time_ms: float
    ) -> ProfileResult:
        """分析性能"""
        # 简化的性能分析
        # TODO: 使用 TensorFlow Profiler 获取详细的层级信息

        layer_times = {}
        ops_count = {
            'Conv2D': 0,
            'MatMul': 0,
            'DnnoptConv2D': 0,
            'DnnoptMatMul': 0,
        }

        # 尝试从计算图中获取算子统计
        try:
            graph_def = self.concrete_func.graph.as_graph_def()
            for node in graph_def.node:
                if node.op in ops_count:
                    ops_count[node.op] += 1
        except:
            pass

        return ProfileResult(
            total_time_ms=total_time_ms,
            layer_times=layer_times,
            ops_count=ops_count,
            dnnopt_speedup=None  # 需要对比测试才能得出
        )

    def benchmark(
        self,
        inputs: Union[Dict[str, np.ndarray], np.ndarray],
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            inputs: 输入数据
            iterations: 迭代次数
            warmup: 预热次数

        Returns:
            性能统计信息
        """
        self.log(f"基准测试: {iterations} 次迭代, {warmup} 次预热")

        # 准备输入
        if isinstance(inputs, np.ndarray) or isinstance(inputs, tf.Tensor):
            if len(self.input_names) == 1:
                inputs = {self.input_names[0]: inputs}
            else:
                raise ValueError(f"模型需要 {len(self.input_names)} 个输入")

        # 转换为 Tensor (只转换一次)
        tensor_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                tensor_inputs[name] = tf.constant(value)
            else:
                tensor_inputs[name] = value

        # 预热
        for _ in range(warmup):
            _ = self.concrete_func(**tensor_inputs)

        # 基准测试
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self.concrete_func(**tensor_inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        # 统计
        latencies = np.array(latencies)
        stats = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': float(1000.0 / np.mean(latencies)),
            'backend': self._backend,
        }

        self.log(f"平均延迟: {stats['mean_ms']:.2f} ms")
        self.log(f"吞吐量: {stats['throughput_fps']:.2f} FPS")

        return stats

    def compare_backends(
        self,
        inputs: Union[Dict[str, np.ndarray], np.ndarray],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        对比不同后端的性能 (仅用于验证)

        注意: 此方法需要在 DNN-Opt 可用时运行，
        然后禁用 DNN-Op 再次运行进行对比。
        """
        self.log("警告: compare_backends 需要手动运行两次，分别启用/禁用 DNN-Opt")

        stats = self.benchmark(inputs, iterations)
        return stats


# ============================================================================
# 批量推理引擎
# ============================================================================

class BatchInferenceEngine:
    """批量推理引擎，支持动态批处理"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        max_queue_size: int = 100,
        timeout_ms: float = 100.0
    ):
        """
        初始化批量推理引擎

        Args:
            model_path: 模型路径
            batch_size: 批处理大小
            max_queue_size: 最大队列大小
            timeout_ms: 超时时间 (毫秒)
        """
        self.engine = DnnoptInferenceEngine(model_path)
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.timeout_ms = timeout_ms

    def infer(self, inputs: np.ndarray) -> np.ndarray:
        """
        执行推理 (自动批处理)

        Args:
            inputs: 输入数组，形状为 [N, ...]

        Returns:
            输出数组
        """
        n = inputs.shape[0]
        outputs = []

        for i in range(0, n, self.batch_size):
            batch = inputs[i:i + self.batch_size]
            result = self.engine.run(batch)
            outputs.append(result.output)

        return np.concatenate(outputs, axis=0)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'DnnoptInferenceEngine',
    'BatchInferenceEngine',
    'InferenceResult',
    'ProfileResult',
]
