#include "caffe2/operators/stop_gradient.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(StopGradient, StopGradientOp<CPUContext>);

// TODO(jiayq): Add example to the doc string.
OPERATOR_SCHEMA(StopGradient)
    .NumInputs(1, 1)
    .NumOutputs(1, 1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
StopGradient is a helper operator that does no actual numerical computation,
and in the gradient computation phase stops the gradient from being computed
through it.
)DOC");

NO_GRADIENT(StopGradient);
}  // namespace caffe2
