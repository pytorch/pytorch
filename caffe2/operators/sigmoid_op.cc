#include "caffe2/operators/sigmoid_op.h"

namespace caffe2 {

template <>
template <typename T>
bool SigmoidFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorArrayMap<T>(Y, N) = 1. / (1. + (-X_arr).exp());
  return true;
}

REGISTER_CPU_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SigmoidFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Sigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor")
    .InheritOnnxSchema("Sigmoid");
// Input: Y, dY, output: dX
OPERATOR_SCHEMA(SigmoidGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
SigmoidGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the sigmoid function.
)DOC");

} // namespace caffe2
