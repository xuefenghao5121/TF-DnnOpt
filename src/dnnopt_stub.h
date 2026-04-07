// dnnopt_stub.h
// DNN-Opt API Definitions (from actual library)
//
// This file documents the actual DNN-Opt API for reference.
// Include the actual headers from DNN-Opt library when available.
//
// DNN-Opt header locations (verified):
//   - dnnopt/conv/conv.h     (Conv2D operations)
//   - dnnopt/gemm/gemm.h     (MatMul/GEMM operations)
//   - dnnopt/arm_hwcaps.h    (ARM hardware capabilities detection)

#ifndef DNNOPT_STUB_H_
#define DNNOPT_STUB_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dnnopt {

// ============================================================================
// Hardware Capabilities (from arm_hwcaps.h)
// ============================================================================

enum HwCap : uint64_t {
    kNone       = 0,
    kNEON       = 1ULL << 0,   // ASIMD (always on AArch64)
    kFP16       = 1ULL << 1,   // Half-precision float
    kDotProd    = 1ULL << 2,   // SDOT/UDOT (ARMv8.2 DotProd)
    kSVE        = 1ULL << 3,   // Scalable Vector Extension
    kSVE2       = 1ULL << 4,   // SVE2 (ARMv9)
    kBF16       = 1ULL << 5,   // BFloat16 (BFMMLA, BFDOT)
    kI8MM       = 1ULL << 6,   // Int8 Matrix Multiply (SMMLA/UMMLA)
    kSME        = 1ULL << 7,   // Scalable Matrix Extension
    kSME2       = 1ULL << 8,   // SME2
    kSVEBF16    = 1ULL << 9,   // SVE BF16 instructions
    kSVEI8MM    = 1ULL << 10,  // SVE I8MM instructions
    kFRINT      = 1ULL << 11,  // FRINT instructions
    kAES        = 1ULL << 12,  // AES crypto extension
    kSHA256     = 1ULL << 13,  // SHA-256 extension
    kAtomics    = 1ULL << 14,  // LSE atomics (ARMv8.1)
};

struct CacheInfo {
    uint32_t size_bytes = 0;
    uint32_t line_size  = 0;
    uint32_t ways       = 0;
    uint32_t sets       = 0;
};

struct CoreCluster {
    uint32_t first_cpu = 0;
    uint32_t count = 0;
    uint32_t max_freq_khz = 0;
    bool is_big = true;
};

struct CoreTopology {
    std::vector<CoreCluster> clusters;
    uint32_t big_cores = 0;
    uint32_t little_cores = 0;
    bool is_heterogeneous = false;
};

struct ArmHwProfile {
    uint32_t    implementer = 0;
    uint32_t    part_number = 0;
    uint32_t    variant     = 0;
    uint32_t    revision    = 0;
    std::string cpu_name;
    uint32_t    num_cores   = 0;
    uint32_t    freq_mhz    = 0;
    uint64_t    hwcaps = kNone;
    uint32_t    sve_vector_bits = 0;
    CacheInfo   l1d;
    CacheInfo   l1i;
    CacheInfo   l2;
    CacheInfo   l3;
    CoreTopology topology;
    double      fp32_gflops_per_core = 0.0;
    double      bf16_gflops_per_core = 0.0;
    double      int8_gops_per_core   = 0.0;

    bool has(HwCap cap) const { return (hwcaps & cap) != 0; }
};

// Detect hardware capabilities
const ArmHwProfile& detect_arm_hwcaps();

// ============================================================================
// Conv2D Operations (from conv/conv.h)
// ============================================================================

struct Conv2DParams {
    int N;             // Batch size
    int IC, IH, IW;   // Input: channels, height, width
    int OC;            // Output channels
    int KH, KW;        // Kernel height, width
    int stride_h, stride_w;
    int pad_h, pad_w;
    int OH_val = 0;    // Pre-computed output height (if 0, will compute)
    int OW_val = 0;    // Pre-computed output width (if 0, will compute)

    int OH() const { return OH_val > 0 ? OH_val : (IH + 2 * pad_h - KH) / stride_h + 1; }
    int OW() const { return OW_val > 0 ? OW_val : (IW + 2 * pad_w - KW) / stride_w + 1; }
};

enum class ConvPostOp {
    kNone,       // No post-op
    kRelu,       // max(0, x)
    kRelu6,      // min(6, max(0, x))
    kBiasRelu,   // max(0, x + bias)
};

// FP32 Conv2D with im2col + optimized GEMM
void conv2d_fp32(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op = ConvPostOp::kNone);

// ============================================================================
// GEMM Operations (from gemm/gemm.h)
// ============================================================================

enum class GemmAlgo {
    kAuto,        // Automatic selection
    kNaive,       // Scalar reference
    kNeonFp32,    // NEON 8x12 FP32 microkernel
    kBf16Bfmmla,  // BF16 BFMMLA microkernel
    kInt8Smmla,   // INT8 SMMLA microkernel
    kInt8Sdot,    // INT8 SDOT microkernel
    kSveFp32,     // SVE FP32 microkernel
    kSveBf16,     // SVE BF16 microkernel
    kSveInt8,     // SVE INT8 microkernel
    kSmeFp32,     // SME FP32 microkernel
};

// FP32 GEMM: C = alpha * A * B + beta * C
void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// FP32 GEMM with explicit algorithm choice
void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc,
               GemmAlgo algo);

// BF16 GEMM: input/output FP32, internal compute BF16
void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// INT8 GEMM: input/output FP32, internal compute INT8
void gemm_int8(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// Thread control
void gemm_set_num_threads(int n);
int gemm_get_num_threads();

}  // namespace dnnopt

#endif  // DNNOPT_STUB_H_
