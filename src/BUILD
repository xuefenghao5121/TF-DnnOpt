# TensorFlow Custom Op Build File
#
# Build rules for DNN-Opt TensorFlow Custom Ops.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

package(default_visibility = ["//visibility:public"])

# ============================================================================
# DNN-Opt Stub Library (for x86_64 testing without ARM hardware)
# ============================================================================

cc_library(
    name = "dnnopt_stub",
    srcs = ["dnnopt_stub_impl.cpp"],
    hdrs = ["dnnopt_stub.h"],
    copts = ["-std=c++17", "-O2"],
    visibility = ["//visibility:public"],
)

# ============================================================================
# DNN-Opt Custom Ops Shared Library
# ============================================================================

# Custom Op library for TensorFlow
# This creates libdnnopt_ops.so that can be loaded via tf.load_op_library()

cc_binary(
    name = "libdnnopt_ops.so",
    srcs = [
        "dnnopt_conv2d_op.cc",
        "dnnopt_matmul_op.cc",
    ],
    deps = [
        ":dnnopt_stub",
        # TensorFlow headers - these come from the TensorFlow installation
        # When building manually, use tensorflow.sysconfig.get_compile_flags()
    ],
    copts = [
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fopenmp",
    ],
    linkopts = [
        "-shared",
        "-fopenmp",
        "-lpthread",
        "-ldl",
    ],
    visibility = ["//visibility:public"],
)

# ============================================================================
# Individual Op Libraries (for testing)
# ============================================================================

cc_library(
    name = "dnnopt_conv2d_op",
    srcs = ["dnnopt_conv2d_op.cc"],
    hdrs = [],
    deps = [
        "@dnnopt//:dnnopt_headers",
    ],
    copts = ["-std=c++17", "-O2", "-fPIC"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dnnopt_matmul_op",
    srcs = ["dnnopt_matmul_op.cc"],
    hdrs = [],
    deps = [
        "@dnnopt//:dnnopt_headers",
    ],
    copts = ["-std=c++17", "-O2", "-fPIC"],
    visibility = ["//visibility:public"],
)
