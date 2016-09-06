#include "caffe2/sgd/iter_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Iter, IterOp<CPUContext>);
REGISTER_CPU_OPERATOR(AtomicIter, AtomicIterOp<CPUContext>);

OPERATOR_SCHEMA(Iter)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.
)DOC");

OPERATOR_SCHEMA(AtomicIter)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{1, 0}})
    .SetDoc(R"DOC(
Similar to Iter, but takes a mutex as the first input to make sure that
updates are carried out atomically. This can be used in e.g. Hogwild sgd
algorithms.
)DOC")
    .Input(0, "mutex", "The mutex used to do atomic increment.")
    .Input(1, "iter", "The iter counter as an int64_t TensorCPU.");

NO_GRADIENT(Iter);
NO_GRADIENT(AtomicIter);
}
}  // namespace caffe2
