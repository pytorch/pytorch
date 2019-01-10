#include "caffe2/operators/elu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool EluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  // Otherwise inplace gradient and Elu dosen't make sense.
  CAFFE_ENFORCE_GE(alpha_, 0);
  Y->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();
  ConstEigenVectorArrayMap<float> Xvec(Xdata, X.size());
  EigenVectorArrayMap<float> Yvec(Ydata, Y->size());
  Yvec = Xvec.cwiseMax(0.f) + (alpha_ * (Xvec.exp() - 1.0f)).cwiseMin(0.f);
  return true;
}

template <>
bool EluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  dXvec = (Yvec > 0).select(dYvec, dYvec * (Yvec + alpha_));
  return true;
}

REGISTER_CPU_OPERATOR(Elu, EluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(EluGradient, EluGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Elu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor")
    .InheritOnnxSchema("Elu");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(EluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
EluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetEluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Elu, GetEluGradient);

} // namespace caffe2
