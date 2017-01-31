#include "allreduce_ops.h"

namespace caffe2 {
namespace fbcollective {
namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allreduce,
    FBCOLLECTIVE,
    AllreduceOp<float, CPUContext>);
}
} // namespace fbcollective
} // namespace caffe2
