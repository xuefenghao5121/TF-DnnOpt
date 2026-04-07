// dnnopt_stub_impl.cpp
// DNN-Opt Stub Implementation for x86_64 testing
//
// This provides stub implementations for DNN-Opt functions
// to test TensorFlow Custom Op compilation on non-ARM platforms.

#include "dnnopt_stub.h"
#include <cstring>
#include <cmath>

namespace dnnopt {

// ============================================================================
// Hardware Capabilities Stub
// ============================================================================

static ArmHwProfile g_hw_profile;

const ArmHwProfile& detect_arm_hwcaps() {
    if (g_hw_profile.num_cores == 0) {
        g_hw_profile.cpu_name = "x86_64_stub";
        g_hw_profile.num_cores = 1;
        g_hw_profile.hwcaps = kNEON;  // Pretend we have NEON for testing
        g_hw_profile.fp32_gflops_per_core = 10.0;
    }
    return g_hw_profile;
}

// ============================================================================
// GEMM Stub Implementation
// ============================================================================

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    // Simple naive GEMM for testing
    // Note: TensorFlow doesn't zero-initialize output tensors,
    // so we must handle beta=0 specially (0 * NaN = NaN)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (beta == 0.0f) {
                C[i * ldc + j] = alpha * sum;
            } else {
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
    }
}

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc,
               GemmAlgo algo) {
    // Ignore algo for stub
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    // Fall back to FP32 for stub
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_int8(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    // Fall back to FP32 for stub
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_set_num_threads(int n) {
    // Stub: ignore
}

int gemm_get_num_threads() {
    return 1;
}

// ============================================================================
// Conv2D Stub Implementation
// ============================================================================

void conv2d_fp32(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op) {
    // Simple im2col + GEMM implementation for testing
    int OH = p.OH();
    int OW = p.OW();

    // Initialize output with bias
    for (int n = 0; n < p.N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int oc = 0; oc < p.OC; ++oc) {
                    float val = (bias != nullptr) ? bias[oc] : 0.0f;
                    output[n * OH * OW * p.OC + oh * OW * p.OC + ow * p.OC + oc] = val;
                }
            }
        }
    }

    // Simple convolution (naive, for testing only)
    for (int n = 0; n < p.N; ++n) {
        for (int oc = 0; oc < p.OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < p.KH; ++kh) {
                        for (int kw = 0; kw < p.KW; ++kw) {
                            for (int ic = 0; ic < p.IC; ++ic) {
                                int ih = oh * p.stride_h - p.pad_h + kh;
                                int iw = ow * p.stride_w - p.pad_w + kw;

                                if (ih >= 0 && ih < p.IH && iw >= 0 && iw < p.IW) {
                                    int input_idx = n * p.IH * p.IW * p.IC +
                                                   ih * p.IW * p.IC +
                                                   iw * p.IC + ic;
                                    int filter_idx = oc * p.KH * p.KW * p.IC +
                                                    kh * p.KW * p.IC +
                                                    kw * p.IC + ic;
                                    sum += input[input_idx] * filter[filter_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * OH * OW * p.OC + oh * OW * p.OC + ow * p.OC + oc;
                    output[out_idx] += sum;

                    // Apply post-op
                    if (post_op == ConvPostOp::kRelu) {
                        output[out_idx] = std::max(0.0f, output[out_idx]);
                    } else if (post_op == ConvPostOp::kRelu6) {
                        output[out_idx] = std::min(6.0f, std::max(0.0f, output[out_idx]));
                    }
                }
            }
        }
    }
}

}  // namespace dnnopt
