#include "barrier_ops.h"

namespace caffe2 {
namespace gloo {
namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Barrier, GLOO, BarrierOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
