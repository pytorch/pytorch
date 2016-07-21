#include "caffe2/sgd/iter_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Iter, IterOp<CPUContext>);

OPERATOR_SCHEMA(Iter)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.
)DOC");

NO_GRADIENT(Iter);
}
}  // namespace caffe2
