#include "caffe2/operators/thresholded_relu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool ThresholdedReluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.size());
  EigenVectorArrayMap<float> Yvec(Y->mutable_data<float>(), Y->size());
  Yvec = (Xvec > alpha_).select(Xvec, 0.f);
  /* Naive implementation
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Xdata[i] -= alpha_;
    Ydata[i] = std::max(Xdata[i], 0.0f);
  }
  */
  return true;
}

template <>
bool ThresholdedReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * Yvec.cwiseSign();
  /* Non vectorized implementation
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
  }
  */
  return true;
}

REGISTER_CPU_OPERATOR(ThresholdedRelu, ThresholdedReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ThresholdedReluGradient,
    ThresholdedReluGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(ThresholdedRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<2>)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
ThresholdedRelu takes one input data (Tensor) and produces one output data
(Tensor) where the rectified linear function, y = x for x > alpha, y = 0
otherwise, is applied to the tensor elementwise.
)DOC")
    .Arg("alpha", "(float) defaults to 1.0.")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(ThresholdedReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
ThresholdedReluGradient takes both Y and dY and uses this to update dX
according to the chain rule and derivatives of the rectified linear function.
)DOC");

class GetThresholdedReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ThresholdedRelu, GetThresholdedReluGradient);

} // namespace caffe2
