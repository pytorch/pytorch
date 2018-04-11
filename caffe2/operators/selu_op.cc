#include "caffe2/operators/selu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool SeluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.size());
  EigenVectorArrayMap<float> Yvec(Y->mutable_data<float>(), Y->size());
  Yvec = lambda_ * (Xvec > 0).select(Xvec, (alpha_ * Xvec.exp() - alpha_));
  return true;
}

template <>
bool SeluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  ConstEigenVectorArrayMap<float> Yvec(Y.data<float>(), Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dY.data<float>(), dY.size());
  EigenVectorArrayMap<float> dXvec(dX->mutable_data<float>(), dX->size());

  const float la = lambda_ * alpha_;
  dXvec = (Yvec > 0).select(lambda_ * dYvec, dYvec * (Yvec + la));
  return true;
}

REGISTER_CPU_OPERATOR(Selu, SeluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SeluGradient, SeluGradientOp<float, CPUContext>);

// Input: X; output: Y
OPERATOR_SCHEMA(Selu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function, y = scale*(alpha_*e^x-alpha_ if x < 0 else x),
is applied to the tensor elementwise.
)DOC")
    .Arg(
        "alpha",
        "(float) default to 1.6732~; affects the activation function itself. "
        "This should go with the weight initialization in the paper. "
        " See https://arxiv.org/abs/1706.02515 ")
    .Arg(
        "scale",
        "(float) default to 1.0507~; affects the activation function itself.")
    .Input(0, "X", "input tensor")
    .Output(0, "Y", "input tensor")
    .InheritOnnxSchema("Selu");

// Input: Y, dY; output: dX
OPERATOR_SCHEMA(SeluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
SeluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the selu function.
)DOC")
    .Arg(
        "alpha",
        "(float) default to 1.6732~; affects the activation function itself."
        "This should go with the weight initialization in the paper. "
        " See https://arxiv.org/abs/1706.02515 ")
    .Arg(
        "scale",
        "(float) default to 1.0507~; affects the activation function itself.")
    .Input(0, "Y", "input tensor")
    .Input(1, "dY", "input tensor");

class GetSeluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Selu, GetSeluGradient);

} // namespace caffe2
