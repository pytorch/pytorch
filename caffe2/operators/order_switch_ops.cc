#include "caffe2/operators/order_switch_ops.h"

namespace caffe2 {

template <>
bool NHWC2NCHWOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  CAFFE_ENFORCE_GE(ndim, 3);
  const int N = X.dim32(0), C = X.dim32(ndim - 1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  Y_dims[1] = C;
  int image_size = 1;
  for (auto i = 2; i < ndim; ++i) {
    Y_dims[i] = X.dim32(i - 1);
    image_size *= Y_dims[i];
  }
  Y->Resize(Y_dims);
  if (X.size() <= 0) {
    return true;
  }

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  std::array<int, 2> dims = {image_size, C};
  std::array<int, 2> axes = {1, 0};
  for (int n = 0; n < N; ++n) {
    math::Transpose(
        2,
        dims.data(),
        axes.data(),
        Xdata + n * image_size * C,
        Ydata + n * image_size * C,
        &context_);
  }
  return true;
}

template <>
bool NCHW2NHWCOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  CAFFE_ENFORCE_GE(X.ndim(), 3);
  const int N = X.dim32(0), C = X.dim32(1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  int image_size = 1;
  for (auto i = 1; i < ndim - 1; ++i) {
    Y_dims[i] = X.dim32(i + 1);
    image_size *= Y_dims[i];
  }
  Y_dims[ndim - 1] = C;
  Y->Resize(Y_dims);
  if (X.size() <= 0) {
    return true;
  }

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  std::array<int, 2> dims = {C, image_size};
  std::array<int, 2> axes = {1, 0};
  for (int n = 0; n < N; ++n) {
    math::Transpose(
        2,
        dims.data(),
        axes.data(),
        Xdata + n * image_size * C,
        Ydata + n * image_size * C,
        &context_);
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
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NHWC2NCHW must be >= 3 dimensional");
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims(0));
      out[0].add_dims(in[0].dims(in[0].dims_size() - 1));
      for (auto i = 1; i < in[0].dims_size() - 1; ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      return out;
    })
    .SetDoc(R"DOC(
The operator switches the order of data in a tensor from NHWC- sample index N,
height H, width H and channels C, to the NCHW order (this is for 2D images).
In general, this operator switches the order of data in a tensor from N H_1 ...
H_k C to N C H_1 ... H_k for k-dimensional features, and currently supports
k=1, 2, and 3.
)DOC")
    .Input(0, "data", "The input data (Tensor) in the NHWC order.")
    .Output(0, "output", "The output tensor (Tensor) in the NCHW order.");

OPERATOR_SCHEMA(NCHW2NHWC)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*unused*/ /*def*/,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NCHW2NHWC must be >= 3 dimensional");
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims(0));
      for (auto i = 2; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].add_dims(in[0].dims(1));
      return out;
    })
    .SetDoc(R"DOC(
The operator switches the order of data in a tensor from NCHW- sample index N,
channels C, height H and width W, to the NHWC order (this is for 2D images).
In general, this operator switches the order of data in a tensor from N C H_1
... H_k to N H_1 ... H_k C for k-dimensional features, and currently supports
k=1, 2, and 3.
)DOC")
    .Input(0, "data", "The input data (Tensor) in the NCHW order.")
    .Output(0, "output", "The output tensor (Tensor) in the NHWC order.");

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
