"""
DNN-Opt TensorFlow Integration

高性能 ARM 平台 TensorFlow 推理优化库。

无感知使用:
    import dnnopt_tensorflow  # 自动替换 TensorFlow 算子
    import tensorflow as tf
    # 之后的代码完全不变

显式使用:
    from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul, DnnoptConv2D
    from inference_engine import DnnoptInferenceEngine
    from model_converter import convert_savedmodel
"""

# 无感知集成 - Monkey Patching
from .dnnopt_tensorflow import (
    enable as enable_dnnopt,
    disable as disable_dnnopt,
    is_patched,
    is_dnnopt_available,
    apply_patch,
)

# 显式 API
from .dnnopt_ops import (
    dnnopt_conv2d,
    dnnopt_matmul,
    dnnopt_batch_matmul,
    DnnoptConv2D,
    get_dnnopt_error,
    reload_dnnopt_ops,
)

from .inference_engine import (
    DnnoptInferenceEngine,
    BatchInferenceEngine,
    InferenceResult,
    ProfileResult,
)

from .model_converter import (
    ModelConverter,
    ConversionStats,
    convert_savedmodel,
    convert_keras_model,
    transpose_conv2d_weights,
)

__version__ = '0.1.0'

__all__ = [
    # 无感知集成
    'enable_dnnopt',
    'disable_dnnopt',
    'is_patched',
    'is_dnnopt_available',
    'apply_patch',

    # 显式 Ops
    'dnnopt_conv2d',
    'dnnopt_matmul',
    'dnnopt_batch_matmul',
    'DnnoptConv2D',

    # 推理引擎
    'DnnoptInferenceEngine',
    'BatchInferenceEngine',
    'InferenceResult',
    'ProfileResult',

    # 模型转换
    'ModelConverter',
    'ConversionStats',
    'convert_savedmodel',
    'convert_keras_model',
    'transpose_conv2d_weights',

    # 工具
    'get_dnnopt_error',
    'reload_dnnopt_ops',
]
