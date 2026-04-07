"""
模型转换工具

将 TensorFlow SavedModel 中的 Conv2D/MatMul 算子替换为 DNN-Opt 优化版本。

使用方法:
    from model_converter import convert_savedmodel

    convert_savedmodel(
        input_dir='path/to/original_model',
        output_dir='path/to/optimized_model'
    )
"""

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2


@dataclass
class ConversionStats:
    """转换统计信息"""
    total_nodes: int = 0
    conv2d_nodes: int = 0
    matmul_nodes: int = 0
    bias_add_nodes: int = 0
    relu_nodes: int = 0
    converted_conv2d: int = 0
    converted_matmul: int = 0
    fused_relu: int = 0
    weight_transposed: int = 0


class ModelConverter:
    """
    TensorFlow 模型转换器

    将标准 TensorFlow 算子转换为 DNN-Opt 优化版本：
    - Conv2D -> DnnoptConv2D
    - MatMul -> DnnoptMatMul
    - 支持融合 BiasAdd + Relu
    """

    # 可替换的算子类型
    REPLACEABLE_OPS = {
        'Conv2D',
        'MatMul',
        'BiasAdd',
        'Relu',
        'Relu6',
    }

    def __init__(self, verbose: bool = True):
        """
        初始化转换器

        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.stats = ConversionStats()

    def log(self, msg: str):
        """输出日志"""
        if self.verbose:
            print(f"[ModelConverter] {msg}")

    def convert_savedmodel(
        self,
        input_dir: str,
        output_dir: str,
        signature_key: str = 'serving_default',
        tags: List[str] = None
    ) -> ConversionStats:
        """
        转换 SavedModel

        Args:
            input_dir: 输入模型目录
            output_dir: 输出模型目录
            signature_key: 签名键名
            tags: 模型标签

        Returns:
            转换统计信息
        """
        if tags is None:
            tags = ['serve']

        self.log(f"加载模型: {input_dir}")

        # 加载原始模型
        loaded = tf.saved_model.load(input_dir)

        # 获取 concrete function
        if signature_key in loaded.signatures:
            concrete_func = loaded.signatures[signature_key]
        else:
            # 尝试获取第一个签名
            sig_keys = list(loaded.signatures.keys())
            if sig_keys:
                signature_key = sig_keys[0]
                concrete_func = loaded.signatures[signature_key]
                self.log(f"使用签名: {signature_key}")
            else:
                raise ValueError(f"未找到签名 '{signature_key}'")

        # 获取 GraphDef
        graph_def = concrete_func.graph.as_graph_def()

        # 转换 GraphDef
        self.log("开始转换计算图...")
        converted_graph_def = self.convert_graphdef(graph_def)

        # 保存转换后的模型
        self.log(f"保存模型: {output_dir}")
        self._save_converted_model(converted_graph_def, output_dir, signature_key, tags)

        self.log(f"转换完成!")
        self.log(f"  Conv2D: {self.stats.converted_conv2d}/{self.stats.conv2d_nodes}")
        self.log(f"  MatMul: {self.stats.converted_matmul}/{self.stats.matmul_nodes}")
        self.log(f"  融合 ReLU: {self.stats.fused_relu}")
        self.log(f"  权重转置: {self.stats.weight_transposed}")

        return self.stats

    def convert_graphdef(self, graph_def: graph_pb2.GraphDef) -> graph_pb2.GraphDef:
        """
        转换 GraphDef

        Args:
            graph_def: 原始 GraphDef

        Returns:
            转换后的 GraphDef
        """
        self.stats = ConversionStats()
        self.stats.total_nodes = len(graph_def.node)

        # 构建节点映射
        node_map: Dict[str, node_def_pb2.NodeDef] = {}
        for node in graph_def.node:
            node_map[node.name] = node

            # 统计原始节点
            if node.op == 'Conv2D':
                self.stats.conv2d_nodes += 1
            elif node.op == 'MatMul':
                self.stats.matmul_nodes += 1
            elif node.op == 'BiasAdd':
                self.stats.bias_add_nodes += 1
            elif node.op in ('Relu', 'Relu6'):
                self.stats.relu_nodes += 1

        # 找到可以融合的节点组
        fusion_groups = self._find_fusion_groups(node_map)

        # 创建新的 GraphDef
        converted_graph = graph_pb2.GraphDef()

        # 需要跳过的节点 (已被融合)
        skip_nodes: Set[str] = set()
        for group in fusion_groups:
            if len(group) > 1:
                for node_name in group[1:]:  # 第一个节点保留，其余跳过
                    skip_nodes.add(node_name)

        # 转换每个节点
        for node in graph_def.node:
            if node.name in skip_nodes:
                continue

            if node.name in fusion_groups:
                # 融合节点组
                group = fusion_groups[node.name]
                new_node = self._create_fused_node(node_map, group)
                if new_node:
                    converted_graph.node.append(new_node)
                else:
                    converted_graph.node.append(node)
            elif node.op == 'Conv2D':
                # 转换单独的 Conv2D
                new_node = self._convert_conv2d(node, node_map)
                converted_graph.node.append(new_node)
                self.stats.converted_conv2d += 1
            elif node.op == 'MatMul':
                # 转换单独的 MatMul
                new_node = self._convert_matmul(node, node_map)
                converted_graph.node.append(new_node)
                self.stats.converted_matmul += 1
            else:
                # 保持其他节点不变
                converted_graph.node.append(node)

        return converted_graph

    def _find_fusion_groups(
        self,
        node_map: Dict[str, node_def_pb2.NodeDef]
    ) -> Dict[str, List[str]]:
        """
        找到可以融合的节点组

        融合模式:
        - Conv2D -> BiasAdd -> Relu/Relu6
        - Conv2D -> Relu/Relu6 (无 BiasAdd)
        - MatMul -> BiasAdd

        Returns:
            Dict[首节点名称, [节点名称列表]]
        """
        fusion_groups: Dict[str, List[str]] = {}

        # 构建输出到消费者的映射
        output_consumers: Dict[str, List[str]] = {}
        for name, node in node_map.items():
            for inp in node.input:
                if inp not in output_consumers:
                    output_consumers[inp] = []
                output_consumers[inp].append(name)

        # 查找 Conv2D 开头的融合组
        for name, node in node_map.items():
            if node.op == 'Conv2D':
                group = [name]
                post_op = 'none'

                # 检查是否有 BiasAdd
                consumers = output_consumers.get(name, [])
                bias_add_node = None
                relu_node = None

                for consumer_name in consumers:
                    consumer = node_map.get(consumer_name)
                    if consumer and consumer.op == 'BiasAdd':
                        bias_add_node = consumer_name
                        break

                if bias_add_node:
                    group.append(bias_add_node)
                    # 检查 BiasAdd 后是否有 ReLU
                    bias_consumers = output_consumers.get(bias_add_node, [])
                    for consumer_name in bias_consumers:
                        consumer = node_map.get(consumer_name)
                        if consumer and consumer.op in ('Relu', 'Relu6'):
                            relu_node = consumer_name
                            break
                else:
                    # 检查 Conv2D 后直接跟 ReLU
                    for consumer_name in consumers:
                        consumer = node_map.get(consumer_name)
                        if consumer and consumer.op in ('Relu', 'Relu6'):
                            relu_node = consumer_name
                            break

                if relu_node:
                    group.append(relu_node)

                if len(group) > 1:
                    fusion_groups[name] = group

        return fusion_groups

    def _create_fused_node(
        self,
        node_map: Dict[str, node_def_pb2.NodeDef],
        group: List[str]
    ) -> Optional[node_def_pb2.NodeDef]:
        """
        创建融合节点

        Args:
            node_map: 节点映射
            group: 节点组 [conv2d_name, bias_add_name?, relu_name?]

        Returns:
            新节点
        """
        if not group:
            return None

        first_node = node_map[group[0]]
        if first_node.op != 'Conv2D':
            return None

        # 创建新的 Conv2D 节点
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(first_node)
        new_node.op = 'DnnoptConv2D'

        # 确定 post_op
        post_op = 'none'
        if len(group) >= 3:  # Conv2D + BiasAdd + Relu
            relu_node = node_map[group[2]]
            if relu_node.op == 'Relu':
                post_op = 'relu'
            elif relu_node.op == 'Relu6':
                post_op = 'relu6'
            self.stats.fused_relu += 1
        elif len(group) == 2:
            second_node = node_map[group[1]]
            if second_node.op == 'BiasAdd':
                pass  # post_op 保持 none
            elif second_node.op == 'Relu':
                post_op = 'relu'
                self.stats.fused_relu += 1
            elif second_node.op == 'Relu6':
                post_op = 'relu6'
                self.stats.fused_relu += 1

        # 设置 post_op 属性
        new_node.attr['post_op'].s = post_op.encode('utf-8')

        self.stats.converted_conv2d += 1

        return new_node

    def _convert_conv2d(
        self,
        node: node_def_pb2.NodeDef,
        node_map: Dict[str, node_def_pb2.NodeDef]
    ) -> node_def_pb2.NodeDef:
        """
        转换单个 Conv2D 节点

        Args:
            node: Conv2D 节点
            node_map: 节点映射

        Returns:
            转换后的节点
        """
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        new_node.op = 'DnnoptConv2D'

        # 添加 post_op 属性 (默认 none)
        new_node.attr['post_op'].s = b'none'

        return new_node

    def _convert_matmul(
        self,
        node: node_def_pb2.NodeDef,
        node_map: Dict[str, node_def_pb2.NodeDef]
    ) -> node_def_pb2.NodeDef:
        """
        转换单个 MatMul 节点

        Args:
            node: MatMul 节点
            node_map: 节点映射

        Returns:
            转换后的节点
        """
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        new_node.op = 'DnnoptMatMul'

        # 添加 precision 属性 (默认 fp32)
        new_node.attr['precision'].s = b'fp32'

        return new_node

    def _save_converted_model(
        self,
        graph_def: graph_pb2.GraphDef,
        output_dir: str,
        signature_key: str,
        tags: List[str]
    ):
        """
        保存转换后的模型

        注意: 这是一个简化实现，直接保存 GraphDef。
        对于完整的 SavedModel，需要更复杂的处理。
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存 GraphDef
        graph_path = os.path.join(output_dir, 'optimized_graph.pb')
        with open(graph_path, 'wb') as f:
            f.write(graph_def.SerializeToString())

        self.log(f"GraphDef 已保存: {graph_path}")

        # 保存转换信息
        import json
        info_path = os.path.join(output_dir, 'conversion_info.json')
        with open(info_path, 'w') as f:
            json.dump({
                'signature_key': signature_key,
                'tags': tags,
                'stats': {
                    'total_nodes': self.stats.total_nodes,
                    'conv2d_nodes': self.stats.conv2d_nodes,
                    'matmul_nodes': self.stats.matmul_nodes,
                    'converted_conv2d': self.stats.converted_conv2d,
                    'converted_matmul': self.stats.converted_matmul,
                    'fused_relu': self.stats.fused_relu,
                }
            }, f, indent=2)


def convert_savedmodel(
    input_dir: str,
    output_dir: str,
    signature_key: str = 'serving_default',
    tags: List[str] = None,
    verbose: bool = True
) -> ConversionStats:
    """
    转换 SavedModel (便捷函数)

    Args:
        input_dir: 输入模型目录
        output_dir: 输出模型目录
        signature_key: 签名键名
        tags: 模型标签
        verbose: 是否输出详细信息

    Returns:
        转换统计信息
    """
    converter = ModelConverter(verbose=verbose)
    return converter.convert_savedmodel(
        input_dir=input_dir,
        output_dir=output_dir,
        signature_key=signature_key,
        tags=tags
    )


def convert_keras_model(
    model: tf.keras.Model,
    output_dir: str,
    verbose: bool = True
) -> ConversionStats:
    """
    转换 Keras 模型

    Args:
        model: Keras 模型
        output_dir: 输出目录
        verbose: 是否输出详细信息

    Returns:
        转换统计信息
    """
    # 创建临时目录保存原始模型
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存为 SavedModel
        tmp_savedmodel_path = os.path.join(tmpdir, 'original')
        model.save(tmp_savedmodel_path, save_format='tf')

        # 转换
        return convert_savedmodel(
            input_dir=tmp_savedmodel_path,
            output_dir=output_dir,
            verbose=verbose
        )


# ============================================================================
# 权重格式转换工具
# ============================================================================

def transpose_conv2d_weights(
    weights: 'numpy.ndarray',
    input_format: str = 'HWIO',
    output_format: str = 'OIHW'
) -> 'numpy.ndarray':
    """
    转换 Conv2D 权重格式

    Args:
        weights: 权重数组
        input_format: 输入格式，如 'HWIO' (TensorFlow 默认)
        output_format: 输出格式，如 'OIHW' (DNN-Opt 要求)

    Returns:
        转换后的权重数组
    """
    import numpy as np

    if input_format == output_format:
        return weights

    # 构建转置轴
    perm = [input_format.index(c) for c in output_format]
    return np.transpose(weights, perm)


def convert_weights_file(
    input_path: str,
    output_path: str,
    input_format: str = 'HWIO',
    output_format: str = 'OIHW'
):
    """
    转换权重文件格式

    Args:
        input_path: 输入权重文件路径
        output_path: 输出权重文件路径
        input_format: 输入格式
        output_format: 输出格式
    """
    import numpy as np

    weights = np.load(input_path)
    converted = transpose_conv2d_weights(weights, input_format, output_format)
    np.save(output_path, converted)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'ModelConverter',
    'ConversionStats',
    'convert_savedmodel',
    'convert_keras_model',
    'transpose_conv2d_weights',
    'convert_weights_file',
]
