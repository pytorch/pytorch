#include "caffe2/operators/clip_op.h"

namespace caffe2 {

template <>
bool ClipOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ResizeLike(X);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::min(std::max(Xdata[i], min_), max_);
  }
  return true;
}

template <>
bool ClipGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = dYdata[i] * (Ydata[i] > min_ && Ydata[i] < max_);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Clip, ClipOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ClipGradient, ClipGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Clip)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively. The clipping
operation can be done in in-place fashion too, where the input and output blobs
are the same.
)DOC")
    .Arg("min", "Minimum value, under which element is replaced by min")
    .Arg("max", "Maximum value, above which element is replaced by max")
    .Input(
        0,
        "input",
        "Input tensor (Tensor<float>) containing elements to be"
        "clipped")
    .Input(
        1,
        "output",
        "Output tensor (Tensor<float>) containing clipped"
        "input elements");

OPERATOR_SCHEMA(ClipGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

class GetClipGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ClipGradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Clip, GetClipGradient);
}  // namespace
}  // namespace caffe2
