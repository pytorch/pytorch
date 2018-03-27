#include "caffe2/operators/softplus_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool SoftplusOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      (ConstEigenVectorMap<float>(X.data<float>(), X.size()).array().exp() +
       1.0f)
          .log();
  return true;
}

template <>
bool SoftplusGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * (1.0 - (-Yvec).exp());
  return true;
}

REGISTER_CPU_OPERATOR(Softplus, SoftplusOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SoftplusGradient, SoftplusGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Softplus)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor")
    .InheritOnnxSchema("Softplus");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(SoftplusGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetSoftplusGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SoftplusGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Softplus, GetSoftplusGradient);

} // namespace caffe2
