#include "caffe2/operators/free_op.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Free, FreeOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Free);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Free)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SameNumberOfOutput()
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(
Frees the content of the blobs. The input and output blobs should be
one-to-one inplace.)DOC");
} // namespace caffe2
