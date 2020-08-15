#include "caffe2/operators/enforce_finite_op.h"

namespace caffe2 {

template <>
template <typename T>
bool EnforceFiniteOp<CPUContext>::DoRunWithType() {
  EnforceOnCPU<T>(Input(0));
  return true;
}

REGISTER_CPU_OPERATOR(EnforceFinite, EnforceFiniteOp<CPUContext>);

OPERATOR_SCHEMA(EnforceFinite)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Raise if there is NaN or Inf values in the input tensor.
)DOC")
    .Input(0, "input", "Input tensor");

SHOULD_NOT_DO_GRADIENT(EnforceFinite);

} // namespace caffe2
