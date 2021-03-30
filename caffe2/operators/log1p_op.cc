#include "caffe2/operators/log1p_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Log1p,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, Log1pFunctor<CPUContext>>);

OPERATOR_SCHEMA(Log1p)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates log1p of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log1p_op.cc
)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor computed as log1p of the input tensor computed, element-wise.")
    .InheritOnnxSchema();

} // namespace caffe2
