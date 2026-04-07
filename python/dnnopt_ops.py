"""
DNN-Opt TensorFlow Operations

提供 DNN-Opt 优化的 TensorFlow 算子封装，支持自动 fallback 到原生 TensorFlow。

使用方法:
    from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul

    # 使用 DNN-Opt Conv2D
    output = dnnopt_conv2d(input, filter, bias, strides=(1,1,1,1), padding='SAME')

    # 使用 DNN-Opt MatMul
    output = dnnopt_matmul(a, b)
"""

import os
import tensorflow as tf
from typing import Optional, Tuple, Union, List

# ============================================================================
# 加载 Custom Op Library
# ============================================================================

_dnnopt_ops = None
_dnnopt_loaded = False
_dnnopt_error = None

def _load_dnnopt_ops():
    """加载 DNN-Opt Custom Op 库"""
    global _dnnopt_ops, _dnnopt_loaded, _dnnopt_error

    if _dnnopt_loaded:
        return _dnnopt_ops is not None

    _dnnopt_loaded = True

    # 尝试多个可能的路径
    possible_paths = [
        # 当前目录
        os.path.join(os.path.dirname(__file__), '..', 'libdnnopt_ops.so'),
        os.path.join(os.path.dirname(__file__), 'libdnnopt_ops.so'),
        # 绝对路径 (从环境变量)
        os.environ.get('DNNOPT_OPS_PATH', ''),
        # 相对于脚本执行目录
        './libdnnopt_ops.so',
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                _dnnopt_ops = tf.load_op_library(path)
                print(f"[DNN-Opt] 成功加载 Custom Op 库: {path}")
                return True
            except Exception as e:
                _dnnopt_error = str(e)
                print(f"[DNN-Opt] 加载失败 ({path}): {e}")
                continue

    print(f"[DNN-Opt] 未找到 Custom Op 库，将使用 TensorFlow 原生算子作为 fallback")
    print(f"[DNN-Opt] 加载错误: {_dnnopt_error}")
    return False


# 在导入时尝试加载
_load_dnnopt_ops()


# ============================================================================
# Conv2D
# ============================================================================

def dnnopt_conv2d(
    input: tf.Tensor,
    filter: tf.Tensor,
    bias: Optional[tf.Tensor] = None,
    strides: Tuple[int, int, int, int] = (1, 1, 1, 1),
    padding: str = 'SAME',
    post_op: str = 'none',
    data_format: str = 'NHWC',
    name: Optional[str] = None
) -> tf.Tensor:
    """
    DNN-Opt 优化的 Conv2D 算子。

    Args:
        input: 输入 tensor，形状为 [N, IH, IW, IC] (NHWC 格式)
        filter: 卷积核，形状为 [KH, KW, IC, OC] (TensorFlow 默认 HWIO 格式)
                或 [OC, KH, KW, IC] (DNN-Opt OIHW 格式)
        bias: 偏置向量，形状为 [OC]，可选
        strides: 步长，格式为 [1, sh, sw, 1]
        padding: 'SAME' 或 'VALID'
        post_op: 后处理操作，可选 'none', 'relu', 'relu6'
        data_format: 数据格式，目前仅支持 'NHWC'
        name: 操作名称

    Returns:
        output: 输出 tensor，形状为 [N, OH, OW, OC]
    """
    if _dnnopt_ops is not None:
        # 使用 DNN-Opt Custom Op
        if bias is None:
            bias = tf.zeros([1])  # DNN-Opt 需要 bias 参数

        return _dnnopt_ops.dnnopt_conv2d(
            input, filter, bias,
            strides=strides,
            padding=padding,
            post_op=post_op,
            data_format=data_format,
            name=name
        )
    else:
        # Fallback to TensorFlow native Conv2D
        output = tf.nn.conv2d(
            input, filter,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name if name else 'conv2d'
        )

        # 添加 bias
        if bias is not None:
            output = tf.nn.bias_add(output, bias, data_format=data_format)

        # 应用 post-op
        if post_op == 'relu':
            output = tf.nn.relu(output)
        elif post_op == 'relu6':
            output = tf.nn.relu6(output)

        return output


# ============================================================================
# MatMul
# ============================================================================

def dnnopt_matmul(
    a: tf.Tensor,
    b: tf.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    precision: str = 'fp32',
    name: Optional[str] = None
) -> tf.Tensor:
    """
    DNN-Opt 优化的 MatMul 算子。

    Args:
        a: 输入矩阵 A，形状为 [M, K] 或 [K, M] (如果 transpose_a=True)
        b: 输入矩阵 B，形状为 [K, N] 或 [N, K] (如果 transpose_b=True)
        transpose_a: 是否转置 A
        transpose_b: 是否转置 B
        precision: 计算精度，可选 'fp32', 'bf16', 'int8'
        name: 操作名称

    Returns:
        output: 输出矩阵，形状为 [M, N]
    """
    if _dnnopt_ops is not None:
        # 使用 DNN-Opt Custom Op
        return _dnnopt_ops.dnnopt_matmul(
            a, b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            precision=precision,
            name=name
        )
    else:
        # Fallback to TensorFlow native MatMul
        return tf.matmul(
            a, b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            name=name
        )


# ============================================================================
# Batch MatMul
# ============================================================================

def dnnopt_batch_matmul(
    x: tf.Tensor,
    y: tf.Tensor,
    adj_x: bool = False,
    adj_y: bool = False,
    name: Optional[str] = None
) -> tf.Tensor:
    """
    DNN-Opt 批量矩阵乘法。

    Args:
        x: 输入 tensor，形状为 [..., M, K] 或 [..., K, M] (如果 adj_x=True)
        y: 输入 tensor，形状为 [..., K, N] 或 [..., N, K] (如果 adj_y=True)
        adj_x: 是否对 x 进行共轭转置
        adj_y: 是否对 y 进行共轭转置
        name: 操作名称

    Returns:
        output: 输出 tensor，形状为 [..., M, N]
    """
    # DNN-Opt 目前不支持 batch matmul，使用 TensorFlow 原生实现
    return tf.linalg.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y, name=name)


# ============================================================================
# 便捷函数
# ============================================================================

def is_dnnopt_available() -> bool:
    """检查 DNN-Opt 是否可用"""
    return _dnnopt_ops is not None


def get_dnnopt_error() -> Optional[str]:
    """获取 DNN-Opt 加载错误信息"""
    return _dnnopt_error


def reload_dnnopt_ops(path: Optional[str] = None) -> bool:
    """
    重新加载 DNN-Opt Custom Op 库。

    Args:
        path: 库文件路径，如果为 None 则自动搜索

    Returns:
        是否加载成功
    """
    global _dnnopt_ops, _dnnopt_loaded, _dnnopt_error

    _dnnopt_ops = None
    _dnnopt_loaded = False
    _dnnopt_error = None

    if path:
        if os.path.exists(path):
            try:
                _dnnopt_ops = tf.load_op_library(path)
                _dnnopt_loaded = True
                print(f"[DNN-Opt] 成功加载 Custom Op 库: {path}")
                return True
            except Exception as e:
                print(f"[DNN-Opt] 加载失败: {e}")
                return False
        else:
            print(f"[DNN-Opt] 文件不存在: {path}")
            return False

    return _load_dnnopt_ops()


# ============================================================================
# 层封装 (用于 Keras)
# ============================================================================

class DnnoptConv2D(tf.keras.layers.Layer):
    """DNN-Opt 优化的 Conv2D 层"""

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = 'SAME',
        post_op: str = 'none',
        use_bias: bool = True,
        kernel_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'zeros',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = (1, strides[0] if isinstance(strides, tuple) else strides,
                        strides[1] if isinstance(strides, tuple) else strides, 1)
        self.padding = padding.upper()
        self.post_op = post_op
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        # input_shape: [N, H, W, C]
        input_channels = input_shape[-1]

        # 创建权重 (HWIO 格式，TensorFlow 默认)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size[0], self.kernel_size[1], input_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        return dnnopt_conv2d(
            inputs, self.kernel, self.bias,
            strides=self.strides,
            padding=self.padding,
            post_op=self.post_op
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides[1:3],
            'padding': self.padding,
            'post_op': self.post_op,
            'use_bias': self.use_bias,
        })
        return config


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'dnnopt_conv2d',
    'dnnopt_matmul',
    'dnnopt_batch_matmul',
    'DnnoptConv2D',
    'is_dnnopt_available',
    'get_dnnopt_error',
    'reload_dnnopt_ops',
]
