#include "caffe2/operators/scale_blobs_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ScaleBlobs, ScaleBlobsOp<CPUContext>);
OPERATOR_SCHEMA(ScaleBlobs)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .AllowInplace([](int, int) { return true; })
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
ScaleBlobs takes one or more input data (Tensor) and produces one
or more output data (Tensor) whose value is the input data tensor
scaled element-wise.
)DOC")
    .Arg("scale", "(float, default 1.0) the scale to apply.");

} // namespace caffe2
