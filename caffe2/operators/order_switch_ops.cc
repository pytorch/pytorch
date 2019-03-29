#include "caffe2/operators/order_switch_ops.h"

#include <string>

namespace caffe2 {

REGISTER_CPU_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, CPUContext>);

OPERATOR_SCHEMA(NHWC2NCHW)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*unused*/ /*def*/,
                                const std::vector<TensorShape>& in) {
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NHWC2NCHW must be >= 3 dimensional");
      std::vector<TensorShape> out(1);
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
                                const std::vector<TensorShape>& in) {
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NCHW2NHWC must be >= 3 dimensional");
      std::vector<TensorShape> out(1);
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

namespace {

class GetNHWC2NCHWGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "NCHW2NHWC",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

class GetNCHW2NHWCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "NHWC2NCHW",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(NHWC2NCHW, GetNHWC2NCHWGradient);
REGISTER_GRADIENT(NCHW2NHWC, GetNCHW2NHWCGradient);

} // namespace caffe2
