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

namespace {

REGISTER_CPU_OPERATOR(LeakyRelu, LeakyReluOp<float, CPUContext>);

OPERATOR_SCHEMA(LeakyRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("alpha", "Coefficient of leakage")
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

} // namespace
} // namespace caffe2
