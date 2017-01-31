#include "broadcast_ops.h"

namespace caffe2 {
namespace fbcollective {
namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Broadcast,
    FBCOLLECTIVE,
    BroadcastOp<float, CPUContext>);
}
} // namespace fbcollective
} // namespace caffe2
