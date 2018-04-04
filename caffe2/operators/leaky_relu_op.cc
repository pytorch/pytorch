#include "caffe2/operators/leaky_relu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool LeakyReluOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  ConstEigenVectorMap<float> Xvec(X.template data<float>(), X.size());
  EigenVectorMap<float> Yvec(Y->template mutable_data<float>(), Y->size());
  Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * alpha_;
  return true;
}

template <>
bool LeakyReluGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(Y);
  CAFFE_ENFORCE_EQ(Y.size(), dY.size());
  ConstEigenVectorMap<float> Yvec(Y.template data<float>(), Y.size());
  ConstEigenVectorMap<float> dYvec(dY.template data<float>(), dY.size());
  EigenVectorMap<float> dXvec(dX->template mutable_data<float>(), dX->size());
  Eigen::VectorXf gtZero = (Yvec.array() >= 0.0f).cast<float>();
  dXvec = dYvec.array() * gtZero.array() -
      dYvec.array() * (gtZero.array() - 1.0f) * alpha_;
  return true;
}

REGISTER_CPU_OPERATOR(LeakyRelu, LeakyReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    LeakyReluGradient,
    LeakyReluGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LeakyRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("alpha", "Coefficient of leakage, default value is 0.01")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<2>)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");
OPERATOR_SCHEMA(LeakyReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .Arg("alpha", "Coefficient of leakage")
    .InheritOnnxSchema("LeakyRelu");

class GetLeakyReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LeakyReluGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LeakyRelu, GetLeakyReluGradient);

} // namespace caffe2
