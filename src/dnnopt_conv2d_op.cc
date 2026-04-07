// dnnopt_conv2d_op.cc
// TensorFlow Custom Op for DNN-Opt Conv2D
//
// 功能:
//   - 调用 DNN-Opt 高性能 Conv2D 实现
//   - 自动检测并转换 Filter 布局
//   - 支持 SAME/VALID padding
//   - 支持 fused post-op (ReLU/ReLU6)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "dnnopt/conv/conv.h"
#include "dnnopt/arm_hwcaps.h"

using namespace tensorflow;

// ============================================================================
// Op 注册
// ============================================================================

REGISTER_OP("DnnoptConv2D")
    .Input("input: float")           // [N, IH, IW, IC] NHWC
    .Input("filter: float")          // [KH, KW, IC, OC] (HWIO) 或 [OC, KH, KW, IC] (OIHW)
    .Input("bias: float")            // [OC] 或空 tensor
    .Output("output: float")         // [N, OH, OW, OC] NHWC
    .Attr("strides: list(int) = [1, 1, 1, 1]")
    .Attr("padding: string = 'SAME'")
    .Attr("post_op: string = 'none'")  // none, relu, relu6
    .Attr("data_format: string = 'NHWC'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // 验证输入 rank
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

        shape_inference::ShapeHandle filter_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));

        // 获取维度
        int64 batch = c->Dim(input_shape, 0);
        int64 ih = c->Dim(input_shape, 1);
        int64 iw = c->Dim(input_shape, 2);
        int64 ic = c->Dim(input_shape, 3);

        // Filter 可能是 HWIO 或 OIHW 格式
        // 我们根据 input channels 来判断
        int64 oc;
        int64 kh, kw;

        // 检查 filter 形状判断布局
        auto filter_dim = c->Dim(filter_shape, 3);
        if (c->ValueKnown(filter_dim) && c->Value(filter_dim) == c->Value(ic)) {
            // HWIO 格式: [KH, KW, IC, OC]
            kh = c->Dim(filter_shape, 0);
            kw = c->Dim(filter_shape, 1);
            oc = c->Dim(filter_shape, 3);
        } else {
            // OIHW 格式: [OC, KH, KW, IC]
            oc = c->Dim(filter_shape, 0);
            kh = c->Dim(filter_shape, 1);
            kw = c->Dim(filter_shape, 2);
        }

        // 获取 strides
        std::vector<int32> strides;
        TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
        int stride_h = strides[1];
        int stride_w = strides[2];

        // 获取 padding
        string padding;
        TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

        // 计算输出空间维度
        int64 oh, ow;
        if (padding == "SAME") {
            oh = (ih + stride_h - 1) / stride_h;
            ow = (iw + stride_w - 1) / stride_w;
        } else {  // VALID
            oh = (ih - c->Value(kh)) / stride_h + 1;
            ow = (iw - c->Value(kw)) / stride_w + 1;
        }

        // 设置输出 shape
        c->set_output(0, c->MakeShape({batch, oh, ow, oc}));
        return Status::OK();
    });

// ============================================================================
// Op Kernel 实现
// ============================================================================

class DnnoptConv2DOp : public OpKernel {
public:
    explicit DnnoptConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
        // 获取属性
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
        OP_REQUIRES_OK(context, context->GetAttr("post_op", &post_op_));
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_));

        // 验证属性
        OP_REQUIRES(context, strides_.size() == 4,
                    errors::InvalidArgument("strides must have 4 elements"));
        OP_REQUIRES(context, strides_[0] == 1 && strides_[3] == 1,
                    errors::InvalidArgument("strides[0] and strides[3] must be 1"));
        OP_REQUIRES(context, data_format_ == "NHWC",
                    errors::InvalidArgument("Only NHWC data format is supported"));

        // 初始化硬件检测 (只执行一次)
        static bool hw_initialized = false;
        if (!hw_initialized) {
            const auto& hw = dnnopt::detect_arm_hwcaps();
            LOG(INFO) << "DNN-Opt Conv2D initialized on: " << hw.cpu_name;
            LOG(INFO) << "Capabilities: NEON=" << hw.has(dnnopt::HwCap::kNEON)
                      << " SVE=" << hw.has(dnnopt::HwCap::kSVE)
                      << " BF16=" << hw.has(dnnopt::HwCap::kBF16);
            hw_initialized = true;
        }
    }

    void Compute(OpKernelContext* context) override {
        // 获取输入 tensors
        const Tensor& input_tensor = context->input(0);
        const Tensor& filter_tensor = context->input(1);
        const Tensor& bias_tensor = context->input(2);

        // 获取输入维度 [N, IH, IW, IC]
        const int N = input_tensor.dim_size(0);
        const int IH = input_tensor.dim_size(1);
        const int IW = input_tensor.dim_size(2);
        const int IC = input_tensor.dim_size(3);

        // 获取 filter 维度并检测布局
        const int fdim0 = filter_tensor.dim_size(0);
        const int fdim1 = filter_tensor.dim_size(1);
        const int fdim2 = filter_tensor.dim_size(2);
        const int fdim3 = filter_tensor.dim_size(3);

        int OC, KH, KW;
        bool filter_is_hwio = (fdim3 == IC);  // HWIO: [KH, KW, IC, OC]

        if (filter_is_hwio) {
            // TensorFlow 默认 HWIO 格式: [KH, KW, IC, OC]
            KH = fdim0;
            KW = fdim1;
            OC = fdim3;
        } else {
            // DNN-Opt OIHW 格式: [OC, KH, KW, IC]
            OC = fdim0;
            KH = fdim1;
            KW = fdim2;
        }

        // 计算 padding 和输出尺寸
        int stride_h = strides_[1];
        int stride_w = strides_[2];
        int pad_h, pad_w;
        int OH, OW;

        if (padding_ == "SAME") {
            OH = (IH + stride_h - 1) / stride_h;
            OW = (IW + stride_w - 1) / stride_w;
            pad_h = std::max(0, (OH - 1) * stride_h + KH - IH);
            pad_w = std::max(0, (OW - 1) * stride_w + KW - IW);
            pad_h /= 2;  // 对称 padding
            pad_w /= 2;
        } else {  // VALID
            OH = (IH - KH) / stride_h + 1;
            OW = (IW - KW) / stride_w + 1;
            pad_h = 0;
            pad_w = 0;
        }

        // 分配输出 tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape({N, OH, OW, OC}), &output_tensor));

        // 如果输出维度为 0，直接返回
        if (N == 0 || OH == 0 || OW == 0 || OC == 0) {
            return;
        }

        // 准备 DNN-Opt 参数
        dnnopt::Conv2DParams params;
        params.N = N;
        params.IC = IC;
        params.IH = IH;
        params.IW = IW;
        params.OC = OC;
        params.KH = KH;
        params.KW = KW;
        params.stride_h = stride_h;
        params.stride_w = stride_w;
        params.pad_h = pad_h;
        params.pad_w = pad_w;

        // 设置 post-op
        dnnopt::ConvPostOp post_op = dnnopt::ConvPostOp::kNone;
        if (post_op_ == "relu") {
            post_op = dnnopt::ConvPostOp::kRelu;
        } else if (post_op_ == "relu6") {
            post_op = dnnopt::ConvPostOp::kRelu6;
        }

        // 获取数据指针
        const float* input_data = input_tensor.flat<float>().data();
        const float* filter_data = filter_tensor.flat<float>().data();
        const float* bias_data = nullptr;
        if (bias_tensor.NumElements() > 0) {
            bias_data = bias_tensor.flat<float>().data();
        }
        float* output_data = output_tensor->flat<float>().data();

        // 如果 filter 是 HWIO 格式，需要转换为 OIHW
        Tensor converted_filter;
        const float* filter_ptr = filter_data;

        if (filter_is_hwio) {
            // 分配转换后的 filter: [OC, KH, KW, IC]
            converted_filter = Tensor(DT_FLOAT, TensorShape({OC, KH, KW, IC}));
            float* converted_data = converted_filter.flat<float>().data();

            // HWIO -> OIHW 转换
            // HWIO: [KH, KW, IC, OC] -> OIHW: [OC, KH, KW, IC]
            for (int oc = 0; oc < OC; ++oc) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        for (int ic = 0; ic < IC; ++ic) {
                            // HWIO index
                            int hwio_idx = kh * KW * IC * OC + kw * IC * OC + ic * OC + oc;
                            // OIHW index
                            int oihw_idx = oc * KH * KW * IC + kh * KW * IC + kw * IC + ic;
                            converted_data[oihw_idx] = filter_data[hwio_idx];
                        }
                    }
                }
            }
            filter_ptr = converted_data;
        }

        // 调用 DNN-Opt Conv2D
        dnnopt::conv2d_fp32(params, input_data, filter_ptr, bias_data,
                            output_data, post_op);
    }

private:
    std::vector<int> strides_;
    string padding_;
    string post_op_;
    string data_format_;
};

// ============================================================================
// 注册 Kernel
// ============================================================================

REGISTER_KERNEL_BUILDER(Name("DnnoptConv2D").Device(DEVICE_CPU), DnnoptConv2DOp);
