#include "caffe2/operators/floor_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Floor, FloorOp<float, CPUContext>);

OPERATOR_SCHEMA(Floor)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor function, y = floor(x), is applied to
the tensor elementwise. Currently supports only float32.
)DOC")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND input tensor");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(Floor);

} // namespace caffe2
