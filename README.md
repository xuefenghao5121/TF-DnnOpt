# TensorFlow + DNN-Opt Integration

将 DNN-Opt 高性能 ARM 优化库集成到 TensorFlow 中，实现 ARM 平台上的最优推理性能。

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

# 设置 DNN-Opt 路径 (可选)
export DNNOPT_DIR=/path/to/dnn-opt

# 编译
chmod +x build.sh
./build.sh
```

### 3. 使用

```python
from python.dnnopt_ops import dnnopt_conv2d, dnnopt_matmul

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

## 项目结构

```
TF-DnnOpt/
├── src/
│   ├── dnnopt_conv2d_op.cc    # Conv2D Custom Op
│   └── dnnopt_matmul_op.cc    # MatMul Custom Op
├── python/
│   ├── __init__.py            # 模块导出
│   ├── dnnopt_ops.py          # Python API
│   ├── inference_engine.py    # 推理引擎
│   └── model_converter.py     # 模型转换工具
├── tests/
│   ├── test_conv2d_correctness.py
│   ├── test_matmul_correctness.py
│   ├── test_inference.py
│   └── run_tests.py
├── benchmarks/
│   ├── benchmark_conv2d.py    # Conv2D 基准测试
│   └── benchmark_model.py     # 模型级基准测试
├── examples/
│   └── resnet50_example.py    # ResNet50 示例
├── build.sh                   # 构建脚本
├── WORKSPACE                  # Bazel 工作区
└── README.md
```

## API 文档

### dnnopt_conv2d

```python
dnnopt_conv2d(
    input,              # [N, H, W, C] NHWC 格式
    filter,             # [KH, KW, IC, OC] TensorFlow 默认格式
    bias=None,          # [OC] 或 None
    strides=(1,1,1,1),  # [1, sh, sw, 1]
    padding='SAME',     # 'SAME' 或 'VALID'
    post_op='none',     # 'none', 'relu', 'relu6'
    data_format='NHWC'
)
```

### dnnopt_matmul

```python
dnnopt_matmul(
    a,                  # [M, K]
    b,                  # [K, N]
    transpose_a=False,
    transpose_b=False,
    precision='fp32'    # 'fp32', 'bf16', 'int8'
)
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
