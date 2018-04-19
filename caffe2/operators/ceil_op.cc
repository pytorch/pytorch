#include "caffe2/operators/ceil_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Ceil, CeilOp<float, CPUContext>);

OPERATOR_SCHEMA(Ceil)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil function, y = ceil(x), is applied to
the tensor elementwise. Currently supports only float32.
)DOC")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND input tensor");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(Ceil);

} // namespace caffe2
