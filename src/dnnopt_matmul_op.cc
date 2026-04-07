// dnnopt_matmul_op.cc
// TensorFlow Custom Op for DNN-Opt MatMul
//
// 功能:
//   - 调用 DNN-Opt 高性能 GEMM 实现
//   - 支持 transpose_a/transpose_b
//   - 自动选择最优算法 (NEON/SVE/BF16/INT8)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "dnnopt_stub.h"

using namespace tensorflow;

// ============================================================================
// Op 注册
// ============================================================================

REGISTER_OP("DnnoptMatMul")
    .Input("a: float")
    .Input("b: float")
    .Output("output: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("precision: string = 'fp32'")  // fp32, bf16, int8
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));

        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b_shape));

        bool transpose_a, transpose_b;
        TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
        TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));

        // 获取维度
        auto m = c->Dim(a_shape, transpose_a ? 1 : 0);
        auto n = c->Dim(b_shape, transpose_b ? 0 : 1);
        auto k_a = c->Dim(a_shape, transpose_a ? 0 : 1);
        auto k_b = c->Dim(b_shape, transpose_b ? 1 : 0);

        // 验证 K 维度匹配
        shape_inference::DimensionHandle k;
        TF_RETURN_IF_ERROR(c->Merge(k_a, k_b, &k));

        // 设置输出 shape
        c->set_output(0, c->MakeShape({m, n}));
        return absl::OkStatus();
    });

// ============================================================================
// Op Kernel 实现
// ============================================================================

class DnnoptMatMulOp : public OpKernel {
public:
    explicit DnnoptMatMulOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
        OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));

        // 验证 precision
        OP_REQUIRES(context,
                    precision_ == "fp32" || precision_ == "bf16" || precision_ == "int8",
                    errors::InvalidArgument("precision must be fp32, bf16, or int8"));

        // 初始化硬件检测 (只执行一次)
        static bool hw_initialized = false;
        if (!hw_initialized) {
            const auto& hw = dnnopt::detect_arm_hwcaps();
            LOG(INFO) << "DNN-Opt MatMul initialized on: " << hw.cpu_name;
            LOG(INFO) << "Capabilities: NEON=" << hw.has(dnnopt::HwCap::kNEON)
                      << " SVE=" << hw.has(dnnopt::HwCap::kSVE)
                      << " BF16=" << hw.has(dnnopt::HwCap::kBF16)
                      << " I8MM=" << hw.has(dnnopt::HwCap::kI8MM);
            hw_initialized = true;
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& a_tensor = context->input(0);
        const Tensor& b_tensor = context->input(1);

        // 获取维度
        int M = a_tensor.dim_size(transpose_a_ ? 1 : 0);
        int K_a = a_tensor.dim_size(transpose_a_ ? 0 : 1);
        int K_b = b_tensor.dim_size(transpose_b_ ? 1 : 0);
        int N = b_tensor.dim_size(transpose_b_ ? 0 : 1);

        // 验证维度匹配
        OP_REQUIRES(context, K_a == K_b,
                    errors::InvalidArgument(
                        "Matrix dimensions incompatible: A is ", M, "x", K_a,
                        ", B is ", K_b, "x", N));

        int K = K_a;

        // 分配输出 tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape({M, N}), &output_tensor));

        // 如果维度为 0，直接返回
        if (M == 0 || N == 0 || K == 0) {
            return;
        }

        // 获取数据指针
        const float* a_data = a_tensor.flat<float>().data();
        const float* b_data = b_tensor.flat<float>().data();
        float* c_data = output_tensor->flat<float>().data();

        // 计算 leading dimensions
        int lda = a_tensor.dim_size(1);  // 原始列数
        int ldb = b_tensor.dim_size(1);  // 原始列数
        int ldc = N;

        // 根据 precision 调用不同的 GEMM
        if (precision_ == "fp32") {
            if (transpose_a_ && transpose_b_) {
                // C = A^T * B^T = (B * A)^T
                // 我们需要计算 B * A 然后转置，或者手动处理
                // 这里简化处理：先转置输入
                Tensor a_t = TransposeMatrix(context, a_tensor);
                Tensor b_t = TransposeMatrix(context, b_tensor);
                dnnopt::gemm_fp32(M, N, K,
                                  1.0f, a_t.flat<float>().data(), K,
                                  b_t.flat<float>().data(), N,
                                  0.0f, c_data, ldc);
            } else if (transpose_a_) {
                // C = A^T * B, A^T is K x M, B is K x N
                // dnnopt expects row-major: we need to compute B^T * A then transpose
                // Or use: C^T = B^T * A, so C = (B^T * A)^T
                // Simpler: transpose A, then compute A_t * B
                Tensor a_t = TransposeMatrix(context, a_tensor);
                dnnopt::gemm_fp32(M, N, K,
                                  1.0f, a_t.flat<float>().data(), K,
                                  b_data, ldb,
                                  0.0f, c_data, ldc);
            } else if (transpose_b_) {
                // C = A * B^T, A is M x K, B is N x K (stored), B^T is K x N
                // Need to transpose B to get K x N
                Tensor b_t = TransposeMatrix(context, b_tensor);
                dnnopt::gemm_fp32(M, N, K,
                                  1.0f, a_data, lda,
                                  b_t.flat<float>().data(), N,
                                  0.0f, c_data, ldc);
            } else {
                // C = A * B, standard GEMM
                dnnopt::gemm_fp32(M, N, K,
                                  1.0f, a_data, lda,
                                  b_data, ldb,
                                  0.0f, c_data, ldc);
            }
        } else if (precision_ == "bf16") {
            // BF16 GEMM (internal BF16 computation)
            if (!transpose_a_ && !transpose_b_) {
                dnnopt::gemm_bf16(M, N, K,
                                  1.0f, a_data, lda,
                                  b_data, ldb,
                                  0.0f, c_data, ldc);
            } else {
                // 对于 transpose 情况，暂时 fallback 到 FP32
                // TODO: 实现 BF16 transpose 支持
                context->SetStatus(errors::Unimplemented(
                    "BF16 MatMul with transpose not yet implemented"));
                return;
            }
        } else if (precision_ == "int8") {
            // INT8 GEMM (internal INT8 computation)
            if (!transpose_a_ && !transpose_b_) {
                dnnopt::gemm_int8(M, N, K,
                                  1.0f, a_data, lda,
                                  b_data, ldb,
                                  0.0f, c_data, ldc);
            } else {
                // 对于 transpose 情况，暂时 fallback 到 FP32
                // TODO: 实现 INT8 transpose 支持
                context->SetStatus(errors::Unimplemented(
                    "INT8 MatMul with transpose not yet implemented"));
                return;
            }
        }
    }

private:
    bool transpose_a_;
    bool transpose_b_;
    string precision_;

    // 转置矩阵 (简单实现，可用于小矩阵)
    // WARNING: 此实现为朴素 O(M*K) 转置，大矩阵性能较差
    // TODO: 使用 NEON/SVE 优化版本或 Strassen 算法
    Tensor TransposeMatrix(OpKernelContext* context, const Tensor& input) {
        int rows = input.dim_size(0);
        int cols = input.dim_size(1);

        Tensor output(DT_FLOAT, TensorShape({cols, rows}));
        const float* in_data = input.flat<float>().data();
        float* out_data = output.flat<float>().data();

        // 对于大矩阵，朴素实现较慢，考虑分块优化
        if (rows * cols > 1024 * 1024) {
            LOG(WARNING) << "Transposing large matrix (" << rows << "x" << cols
                         << ") may be slow. Consider using smaller tiles or "
                         << "pre-transposed weights.";
        }

        // 简单转置: out[j][i] = in[i][j]
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out_data[j * rows + i] = in_data[i * cols + j];
            }
        }

        return output;
    }
};

// ============================================================================
// 注册 Kernel
// ============================================================================

REGISTER_KERNEL_BUILDER(Name("DnnoptMatMul").Device(DEVICE_CPU), DnnoptMatMulOp);
