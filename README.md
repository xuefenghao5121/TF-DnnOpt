# TensorFlow + DNN-Opt Integration

将 DNN-Opt 高性能 ARM 优化库集成到 TensorFlow 中，实现 ARM 平台上的最优推理性能。

**最大特点：用户无需修改任何代码，自动获得性能提升！**

## 性能预期

| 场景 | vs TensorFlow Eigen | vs TensorFlow oneDNN |
|------|---------------------|----------------------|
| ResNet50 推理 | 46% 提升 | 38% 提升 |
| BERT 推理 | 50% 提升 | 41% 提升 |

## 支持平台

- Neoverse N1/N2/V1/V2
- Cortex-A78/X2/X3
- A64FX
- Kunpeng 920

## 快速开始

### 1. 环境要求

- TensorFlow 2.x
- Python 3.7+
- C++17 编译器
- ARM CPU (支持 NEON/SVE)

### 2. 编译

```bash
# 克隆项目
git clone https://github.com/xuefenghao5121/TF-DnnOpt.git
cd TF-DnnOpt

# 编译
chmod +x build.sh
./build.sh
```

### 3. 无感知使用（推荐）

**只需一行代码，无需修改任何 TensorFlow 代码：**

```python
import dnnopt_tensorflow  # 在导入 tensorflow 之前或之后导入
import tensorflow as tf

# 以下代码完全不变，自动使用 DNN-Opt 优化
model = tf.keras.applications.ResNet50(weights='imagenet')
output = model(input_image)  # 自动使用 DNN-Opt Conv2D

# tf.nn.conv2d 自动使用 DNN-Opt
output = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

# tf.matmul 自动使用 DNN-Opt
output = tf.matmul(a, b)
```

**控制选项：**

```python
import dnnopt_tensorflow

# 启用/禁用 DNN-Opt
dnnopt_tensorflow.disable()  # 禁用，恢复原生 TensorFlow
dnnopt_tensorflow.enable()   # 重新启用

# 检查状态
print(dnnopt_tensorflow.is_patched())         # 是否已 patch
print(dnnopt_tensorflow.is_dnnopt_available()) # DNN-Opt 是否可用
```

**环境变量控制：**

```bash
DNNOPT_DISABLE=1   # 禁用 DNN-Opt patching
DNNOPT_VERBOSE=1   # 显示详细日志
```

## 项目结构

```
TF-DnnOpt/
├── src/
│   ├── dnnopt_conv2d_op.cc    # Conv2D Custom Op
│   └── dnnopt_matmul_op.cc    # MatMul Custom Op
├── python/
│   ├── __init__.py            # 模块导出
│   ├── dnnopt_tensorflow.py   # ⭐ 无感知集成 (Monkey Patching)
│   ├── dnnopt_ops.py          # 显式 Python API
│   ├── inference_engine.py    # 推理引擎
│   └── model_converter.py     # 模型转换工具
├── tests/
│   ├── test_conv2d_correctness.py
│   ├── test_matmul_correctness.py
│   └── test_inference.py
├── benchmarks/
│   ├── benchmark_conv2d.py
│   └── benchmark_model.py
├── examples/
│   ├── transparent_usage_example.py  # ⭐ 无感知使用示例
│   └── resnet50_example.py
├── build.sh
├── WORKSPACE
└── README.md
```

## 无感知集成原理

导入 `dnnopt_tensorflow` 后，自动替换以下 TensorFlow API：

| 原生 API | 替换为 |
|----------|--------|
| `tf.nn.conv2d` | DNN-Opt Conv2D |
| `tf.matmul` | DNN-Opt MatMul |
| `tf.keras.layers.Conv2D` | DNN-Opt Conv2D Layer |
| `tf.keras.layers.Dense` | DNN-Opt MatMul Layer |

**自动 Fallback：** 当 DNN-Opt 不支持某些参数（如 dilation、NCHW 格式）时，自动回退到 TensorFlow 原生实现。

## 显式 API（可选）

如果需要显式控制，也可以直接使用 API：

```python
from dnnopt_ops import dnnopt_conv2d, dnnopt_matmul

# Conv2D
output = dnnopt_conv2d(
    input, filter, bias,
    strides=(1, 1, 1, 1),
    padding='SAME',
    post_op='relu'
)

# MatMul
output = dnnopt_matmul(a, b, precision='fp32')
```

## 构建选项

```bash
# 仅编译 DNN-Opt
./build.sh --dnnopt-only

# 仅编译 Custom Op (DNN-Opt 已编译)
./build.sh --ops-only

# 跳过验证
./build.sh --no-verify
```

## 开发阶段

- [x] 阶段 1: 基础设施
- [x] 阶段 2: Conv2D Custom Op
- [x] 阶段 3: MatMul Custom Op
- [x] 阶段 4: Python 封装
- [x] 阶段 5: 模型转换工具
- [x] 阶段 6: 测试与优化

## 测试

### 运行所有测试

```bash
cd tests
python run_tests.py
```

### 运行特定测试

```bash
# Conv2D 正确性测试
python tests/test_conv2d_correctness.py

# MatMul 正确性测试
python tests/test_matmul_correctness.py

# 端到端推理测试
python tests/test_inference.py

# 包含性能测试
python tests/test_matmul_correctness.py --perf
```

### 性能基准测试

```bash
# Conv2D 基准测试
python benchmarks/benchmark_conv2d.py --config resnet --iterations 100

# 模型级基准测试
python benchmarks/benchmark_model.py --model cnn --batch-size 1
```

## 推理引擎使用

```python
from inference_engine import DnnoptInferenceEngine

# 创建推理引擎
engine = DnnoptInferenceEngine('path/to/saved_model')

# 单次推理
result = engine.run(input_data)
print(f"延迟: {result.latency_ms:.2f} ms")

# 基准测试
stats = engine.benchmark(input_data, iterations=100)
print(f"平均延迟: {stats['mean_ms']:.2f} ms")
print(f"吞吐量: {stats['throughput_fps']:.2f} FPS")
```

## 模型转换

```python
from model_converter import convert_savedmodel, convert_keras_model

# 转换 SavedModel
stats = convert_savedmodel(
    input_dir='path/to/original_model',
    output_dir='path/to/optimized_model'
)

# 转换 Keras 模型
model = tf.keras.applications.ResNet50(weights='imagenet')
stats = convert_keras_model(model, output_dir='path/to/optimized_model')
```

## License

MIT
