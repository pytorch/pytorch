#include "caffe2/contrib/gloo/common_world_ops.h"

#include <gloo/transport/tcp/device.h>

namespace caffe2 {
namespace gloo {

template <>
void CreateCommonWorld<CPUContext>::initializeForContext() {
  // Nothing to initialize for CPUContext.
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<CPUContext>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CloneCommonWorld,
    GLOO,
    CloneCommonWorld<CPUContext>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(DestroyCommonWorld, GLOO, DestroyCommonWorld);

} // namespace
} // namespace gloo
} // namespace caffe2
