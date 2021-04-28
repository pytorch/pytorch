#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Accumulate, AccumulateOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Accumulate)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Accumulate operator accumulates the input tensor to the output tensor. If the
output tensor already has the right size, we add to it; otherwise, we first
initialize the output tensor to all zeros, and then do accumulation. Any
further calls to the operator, given that no one else fiddles with the output
in the interim, will do simple accumulations.
Accumulation is done using Axpby operation as shown:
  Y = 1*X + gamma*Y
where X is the input tensor, Y is the output tensor and gamma is the multiplier
argument.
)DOC")
  .Arg("gamma", "(float, default 1.0) Accumulation multiplier")
  .Input(0, "input", "The input tensor that has to be accumulated to the "
         "output tensor. If the output size is not the same as input size, the "
         "output tensor is first reshaped and initialized to zero, and only "
         "then, accumulation is done.")
  .Output(0, "output", "Accumulated output tensor");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Accumulate);
}  // namespace caffe2
