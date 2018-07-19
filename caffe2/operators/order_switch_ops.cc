#include "caffe2/operators/order_switch_ops.h"

namespace caffe2 {

template <>
bool NHWC2NCHWOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE(X.ndim() == 4);
  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
  Y->Resize(N, C, H, W);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          Ydata[((n * C + c) * H + h) * W + w] = *(Xdata++);
        }
      }
    }
  }
  return true;
}

template <>
bool NCHW2NHWCOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE(X.ndim() == 4);
  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  Y->Resize(N, H, W, C);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          Ydata[((n * H + h) * W + w) * C + c] = *(Xdata++);
        }
      }
    }
  }
  return true;
}


REGISTER_CPU_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, CPUContext>);

OPERATOR_SCHEMA(NHWC2NCHW)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*unused*/ /*def*/,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(
          in[0].dims_size(), 4, "Input for NHWC2NCHW must be 4 dimensional");
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims(0));
      out[0].add_dims(in[0].dims(3));
      out[0].add_dims(in[0].dims(1));
      out[0].add_dims(in[0].dims(2));
      return out;
    })
    .SetDoc(R"DOC(
The operator switches the order of data in a tensor from NHWC- sample index N,
height H, width H and channels C, to the NCHW order.
)DOC")
    .Input(0, "data", "The input data (Tensor<float>) in the NHWC order.")
    .Output(
        0,
        "output",
        "The output tensor (Tensor<float>) in the NCHW order.");

OPERATOR_SCHEMA(NCHW2NHWC).NumInputs(1).NumOutputs(1)
  .SetDoc(R"DOC(
The operator switches the order of data in a tensor from NCHW- sample index N,
channels C, height H and width W, to the NHWC order.
)DOC")
  .Input(0, "data", "The input data (Tensor<float>) in the NCHW order.")
  .Output(0, "output", "The output tensor (Tensor<float>) in the NHWC order.");


class GetNHWC2NCHWGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "NCHW2NHWC", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(NHWC2NCHW, GetNHWC2NCHWGradient);

class GetNCHW2NHWCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "NHWC2NCHW", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(NCHW2NHWC, GetNCHW2NHWCGradient);
}  // namespace caffe2
