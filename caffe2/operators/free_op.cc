#include "caffe2/operators/free_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Free, FreeOp<CPUContext>);
SHOULD_NOT_DO_GRADIENT(Free);

OPERATOR_SCHEMA(Free)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SameNumberOfOutput()
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(
Frees the content of the blobs. The input and output blobs should be
one-to-one inplace.)DOC");
} // namespace caffe2
