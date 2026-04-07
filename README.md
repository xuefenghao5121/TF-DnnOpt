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
│   └── dnnopt_ops.py          # Python API
├── tests/
│   └── test_conv2d_correctness.py
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
- [ ] 阶段 5: 模型转换工具
- [ ] 阶段 6: 测试与优化

## License

MIT
