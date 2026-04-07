"""
DNN-Opt TensorFlow Monkey Patching

自动替换 TensorFlow 原生算子为 DNN-Opt 优化版本，用户无需修改任何代码。

使用方法:
    # 在导入 tensorflow 之前或之后导入即可
    import dnnopt_tensorflow
    import tensorflow as tf

    # 之后的代码完全不变，自动使用 DNN-Opt
    output = tf.nn.conv2d(input, filter, strides=(1,1,1,1), padding='SAME')
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, 3)])

启用/禁用:
    import dnnopt_tensorflow
    dnnopt_tensorflow.disable()   # 禁用，恢复原生 TensorFlow
    dnnopt_tensorflow.enable()    # 重新启用 DNN-Opt

环境变量控制:
    DNNOPT_DISABLE=1   # 禁用 DNN-Opt patching
    DNNOPT_VERBOSE=1   # 显示详细日志
"""

import os
import sys
import functools
from typing import Optional, Callable, Any

# ============================================================================
# 全局状态
# ============================================================================

_patched = False
_original_tf_nn_conv2d = None
_original_tf_nn_convolution = None
_original_tf_matmul = None
_original_tf_linalg_matmul = None
_original_keras_conv2d = None
_original_keras_dense = None

_dnnopt_available = False
_dnnopt_ops = None
_verbose = os.environ.get('DNNOPT_VERBOSE', '0') == '1'


def log(msg: str):
    """输出日志"""
    if _verbose:
        print(f"[DNN-Opt Patch] {msg}"")


def is_patched() -> bool:
    """检查是否已应用 patch"""
    return _patched


def is_dnnopt_available() -> bool:
    """检查 DNN-Opt 是否可用"""
    return _dnnopt_available


# ============================================================================
# 加载 DNN-Opt Custom Op
# ============================================================================

def _load_dnnopt_ops():
    """加载 DNN-Opt Custom Op 库"""
    global _dnnopt_ops, _dnnopt_available

    if os.environ.get('DNNOPT_DISABLE', '0') == '1':
        log("DNN-Opt 已通过环境变量禁用")
        return False

    try:
        import tensorflow as tf
    except ImportError:
        log("TensorFlow 未安装")
        return False

    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'libdnnopt_ops.so'),
        os.path.join(os.path.dirname(__file__), 'libdnnopt_ops.so'),
        os.environ.get('DNNOPT_OPS_PATH', ''),
        './libdnnopt_ops.so',
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                _dnnopt_ops = tf.load_op_library(path)
                _dnnopt_available = True
                log(f"成功加载 DNN-Opt Custom Op: {path}")
                return True
            except Exception as e:
                log(f"加载失败 ({path}): {e}")
                continue

    log("未找到 DNN-Opt Custom Op 库，将使用 TensorFlow 原生算子")
    return False


# ============================================================================
# Patch 函数
# ============================================================================

def _create_patched_conv2d(original_conv2d: Callable) -> Callable:
    """创建 patch 后的 tf.nn.conv2d"""

    @functools.wraps(original_conv2d)
    def patched_conv2d(
        input,
        filter,
        strides=None,
        padding=None,
        data_format=None,
        dilations=None,
        name=None,
        **kwargs
    ):
        # 如果 DNN-Opt 不可用，使用原生实现
        if not _dnnopt_available or not _patched:
            return original_conv2d(
                input, filter,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
                **kwargs
            )

        # 转换参数
        if strides is None:
            strides = [1, 1, 1, 1]
        if padding is None:
            padding = 'SAME'
        if data_format is None:
            data_format = 'NHWC'
        if dilations is None:
            dilations = [1, 1, 1, 1]

        # DNN-Opt 目前仅支持 NHWC 和 dilations=[1,1,1,1]
        if data_format != 'NHWC' or dilations != [1, 1, 1, 1]:
            log("参数不支持，fallback 到 TensorFlow")
            return original_conv2d(
                input, filter,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
                **kwargs
            )

        # 使用 DNN-Opt Conv2D
        # 创建零 bias (DNN-Op 当前实现需要)
        import tensorflow as tf
        oc = filter.shape[-1] if hasattr(filter, 'shape') else filter.get_shape()[-1]
        bias = tf.zeros([oc], dtype=input.dtype)

        try:
            result = _dnnopt_ops.dnnopt_conv2d(
                input, filter, bias,
                strides=strides,
                padding=padding,
                post_op='none',
                data_format=data_format,
                name=name
            )
            log(f"使用 DNN-Opt Conv2D: input={input.shape}, filter={filter.shape}")
            return result
        except Exception as e:
            log(f"DNN-Opt 失败，fallback: {e}")
            return original_conv2d(
                input, filter,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations,
                name=name,
                **kwargs
            )

    return patched_conv2d


def _create_patched_matmul(original_matmul: Callable) -> Callable:
    """创建 patch 后的 tf.matmul"""

    @functools.wraps(original_matmul)
    def patched_matmul(
        a,
        b,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
        output_type=None,
        name=None,
        **kwargs
    ):
        # 如果 DNN-Opt 不可用，使用原生实现
        if not _dnnopt_available or not _patched:
            return original_matmul(
                a, b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
                output_type=output_type,
                name=name,
                **kwargs
            )

        # DNN-Opt 仅支持基本矩阵乘法
        if adjoint_a or adjoint_b or a_is_sparse or b_is_sparse or output_type is not None:
            log("参数不支持，fallback 到 TensorFlow")
            return original_matmul(
                a, b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
                output_type=output_type,
                name=name,
                **kwargs
            )

        try:
            result = _dnnopt_ops.dnnopt_matmul(
                a, b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                precision='fp32',
                name=name
            )
            log(f"使用 DNN-Opt MatMul: a={a.shape}, b={b.shape}")
            return result
        except Exception as e:
            log(f"DNN-Opt 失败，fallback: {e}")
            return original_matmul(
                a, b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
                output_type=output_type,
                name=name,
                **kwargs
            )

    return patched_matmul


# ============================================================================
# Keras 层 Patch
# ============================================================================

def _create_patched_keras_conv2d():
    """创建 patch 后的 Keras Conv2D 层"""
    import tensorflow as tf

    class PatchedConv2D(tf.keras.layers.Conv2D):
        """自动使用 DNN-Opt 的 Conv2D 层"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def call(self, inputs):
            if not _dnnopt_available or not _patched:
                return super().call(inputs)

            # 使用 DNN-Opt
            import tensorflow as tf

            # 获取参数
            strides = [1, self.strides[0], self.strides[1], 1]
            padding = self.padding.upper()

            # 处理 dilation
            if self.dilation_rate != (1, 1):
                return super().call(inputs)

            # 获取权重
            kernel = self.kernel
            bias = self.bias if self.use_bias else tf.zeros([self.filters])

            # 确定 post_op
            post_op = 'none'
            if self.activation is not None:
                act_name = self.activation.__name__ if hasattr(self.activation, '__name__') else ''
                if act_name == 'relu':
                    post_op = 'relu'
                elif act_name == 'relu6':
                    post_op = 'relu6'

            try:
                output = _dnnopt_ops.dnnopt_conv2d(
                    inputs, kernel, bias,
                    strides=strides,
                    padding=padding,
                    post_op=post_op,
                    data_format='NHWC'
                )

                # 如果有其他 activation (非 relu/relu6)，单独应用
                if self.activation is not None and post_op == 'none':
                    output = self.activation(output)

                return output
            except Exception as e:
                log(f"DNN-Opt Conv2D 失败，fallback: {e}")
                return super().call(inputs)

    return PatchedConv2D


def _create_patched_keras_dense():
    """创建 patch 后的 Keras Dense 层"""
    import tensorflow as tf

    class PatchedDense(tf.keras.layers.Dense):
        """自动使用 DNN-Opt 的 Dense 层"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def call(self, inputs):
            if not _dnnopt_available or not _patched:
                return super().call(inputs)

            # 使用 DNN-Opt MatMul
            import tensorflow as tf

            kernel = self.kernel
            bias = self.bias if self.use_bias else None

            # 转置 kernel (Keras Dense 使用 [in, out]，MatMul 需要)
            try:
                # inputs @ kernel
                output = _dnnopt_ops.dnnopt_matmul(
                    inputs, kernel,
                    transpose_a=False,
                    transpose_b=False,
                    precision='fp32'
                )

                if bias is not None:
                    output = tf.nn.bias_add(output, bias)

                if self.activation is not None:
                    output = self.activation(output)

                return output
            except Exception as e:
                log(f"DNN-Opt Dense 失败，fallback: {e}")
                return super().call(inputs)

    return PatchedDense


# ============================================================================
# 启用/禁用 Patch
# ============================================================================

def enable():
    """启用 DNN-Opt patching"""
    global _patched

    if _original_tf_nn_conv2d is None:
        log("尚未初始化，请先导入 tensorflow")
        return

    import tensorflow as tf

    # 应用 patch
    tf.nn.conv2d = _create_patched_conv2d(_original_tf_nn_conv2d)
    tf.matmul = _create_patched_matmul(_original_tf_matmul)

    if hasattr(tf.linalg, 'matmul'):
        tf.linalg.matmul = tf.matmul

    # Patch Keras
    if hasattr(tf.keras.layers, 'Conv2D'):
        tf.keras.layers.Conv2D = _create_patched_keras_conv2d()
        log("已 patch tf.keras.layers.Conv2D")

    if hasattr(tf.keras.layers, 'Dense'):
        tf.keras.layers.Dense = _create_patched_keras_dense()
        log("已 patch tf.keras.layers.Dense")

    _patched = True
    log("DNN-Opt patching 已启用")


def disable():
    """禁用 DNN-Opt patching，恢复原生 TensorFlow"""
    global _patched

    if _original_tf_nn_conv2d is None:
        return

    import tensorflow as tf

    # 恢复原始函数
    tf.nn.conv2d = _original_tf_nn_conv2d
    tf.matmul = _original_tf_matmul

    if hasattr(tf.linalg, 'matmul') and _original_tf_linalg_matmul is not None:
        tf.linalg.matmul = _original_tf_linalg_matmul

    if _original_keras_conv2d is not None:
        tf.keras.layers.Conv2D = _original_keras_conv2d

    if _original_keras_dense is not None:
        tf.keras.layers.Dense = _original_keras_dense

    _patched = False
    log("DNN-Opt patching 已禁用，恢复原生 TensorFlow")


def apply_patch():
    """应用 DNN-Opt monkey patching"""
    global _patched, _original_tf_nn_conv2d, _original_tf_matmul
    global _original_tf_linalg_matmul, _original_keras_conv2d, _original_keras_dense

    if _patched:
        log("已经应用过 patch")
        return

    if os.environ.get('DNNOPT_DISABLE', '0') == '1':
        log("DNN-Opt 已通过环境变量禁用")
        return

    # 导入 TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        log("TensorFlow 未安装，无法应用 patch")
        return

    # 保存原始函数
    _original_tf_nn_conv2d = tf.nn.conv2d
    _original_tf_matmul = tf.matmul

    if hasattr(tf.linalg, 'matmul'):
        _original_tf_linalg_matmul = tf.linalg.matmul

    # 保存原始 Keras 层
    if hasattr(tf.keras.layers, 'Conv2D'):
        _original_keras_conv2d = tf.keras.layers.Conv2D

    if hasattr(tf.keras.layers, 'Dense'):
        _original_keras_dense = tf.keras.layers.Dense

    # 加载 DNN-Opt
    _load_dnnopt_ops()

    # 应用 patch
    enable()

    log(f"Patching 完成，DNN-Opt 可用: {_dnnopt_available}")


# ============================================================================
# 自动应用
# ============================================================================

# 在导入此模块时自动应用 patch
apply_patch()


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'enable',
    'disable',
    'is_patched',
    'is_dnnopt_available',
    'apply_patch',
]
