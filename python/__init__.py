"""
DNN-Opt TensorFlow Integration

高性能 ARM 平台 TensorFlow 推理优化库。

使用方法:
    from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul, DnnoptConv2D
    from inference_engine import DnnoptInferenceEngine
    from model_converter import convert_savedmodel
"""

from .dnnopt_ops import (
    dnnopt_conv2d,
    dnnopt_matmul,
    dnnopt_batch_matmul,
    DnnoptConv2D,
    is_dnnopt_available,
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
    # Ops
    'dnnopt_conv2d',
    'dnnopt_matmul',
    'dnnopt_batch_matmul',
    'DnnoptConv2D',

    # Engine
    'DnnoptInferenceEngine',
    'BatchInferenceEngine',
    'InferenceResult',
    'ProfileResult',

    # Converter
    'ModelConverter',
    'ConversionStats',
    'convert_savedmodel',
    'convert_keras_model',
    'transpose_conv2d_weights',

    # Utils
    'is_dnnopt_available',
    'get_dnnopt_error',
    'reload_dnnopt_ops',
]
