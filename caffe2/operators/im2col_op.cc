#include "caffe2/operators/im2col_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Im2Col, Im2ColOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Col2Im, Col2ImOp<float, CPUContext>);

class GetIm2ColGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Col2Im",
        "",
        std::vector<string>{GO(0), I(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Im2Col, GetIm2ColGradient);

class GetCol2ImGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Im2Col", "", std::vector<string>{GO(0)}, std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Col2Im, GetCol2ImGradient);

OPERATOR_SCHEMA(Im2Col)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("The Im2Col operator from Matlab.")
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);
          auto pad = helper.GetSingleArgument<int>("pad", 0);
          auto kernel_h = helper.GetSingleArgument<int>(
              "kernel_h", helper.GetSingleArgument<int>("kernel", 0));
          auto kernel_w = helper.GetSingleArgument<int>(
              "kernel_w", helper.GetSingleArgument<int>("kernel", 0));
          auto dilation_h = helper.GetSingleArgument<int>(
              "dilation_h", helper.GetSingleArgument<int>("dilation", 1));
          auto dilation_w = helper.GetSingleArgument<int>(
              "dilation_w", helper.GetSingleArgument<int>("dilation", 1));
          auto stride_h = helper.GetSingleArgument<int>(
              "stride_h", helper.GetSingleArgument<int>("stride", 1));
          auto stride_w = helper.GetSingleArgument<int>(
              "stride_w", helper.GetSingleArgument<int>("stride", 1));
          auto order = StringToStorageOrder(
              helper.GetSingleArgument<string>("order", "NCHW"));

          const TensorShape& X = in[0];
          int N = 0, C = 0, H = 0, W = 0;
          switch (order) {
            case StorageOrder::NCHW:
              N = X.dims(0);
              C = X.dims(1);
              H = X.dims(2);
              W = X.dims(3);
              break;
            case StorageOrder::NHWC:
              N = X.dims(0);
              H = X.dims(1);
              W = X.dims(2);
              C = X.dims(3);
              break;
            default:
              CAFFE_THROW("Unknown storage order: ", order);
          }

          const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
          const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
          CAFFE_ENFORCE(H >= dkernel_h);
          CAFFE_ENFORCE(W >= dkernel_w);
          const int out_h = (H + 2 * pad - dkernel_h) / stride_h + 1;
          const int out_w = (W + 2 * pad - dkernel_w) / stride_w + 1;

          vector<TensorShape> out(1);
          switch (order) {
            case StorageOrder::NCHW:
              out[0] = CreateTensorShape(
                  vector<int>{N, C * kernel_h * kernel_w, out_h, out_w},
                  TensorProto::FLOAT);
              break;
            case StorageOrder::NHWC:
              out[0] = CreateTensorShape(
                  vector<int>{N, out_h, out_w, kernel_h * kernel_w * C},
                  TensorProto::FLOAT);
              break;
            default:
              CAFFE_THROW("Unknown storage order: ", order);
          }

          return out;
        })
    .Input(0, "X", "4-tensor in NCHW or NHWC.")
    .Output(
        0,
        "Y",
        "4-tensor. For NCHW: N x (C x kH x kW) x outH x outW."
        "For NHWC: N x outH x outW x (kH x kW x C");

OPERATOR_SCHEMA(Col2Im).NumInputs(2).NumOutputs(1);

} // namespace caffe2
