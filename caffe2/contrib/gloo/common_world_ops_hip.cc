#include "caffe2/contrib/gloo/common_world_ops.h"

#include "caffe2/core/hip/context_hip.h"

#include <gloo/hip.h>
#include <gloo/transport/tcp/device.h>

namespace caffe2 {
namespace gloo {

template <>
void CreateCommonWorld<HIPContext>::initializeForContext() {
  static std::once_flag once;
  std::call_once(once, [&]() {
      // This is the first time we call Gloo code for a HIPContext.
      // Share Caffe2 HIP mutex with Gloo.
      ::gloo::HipShared::setMutex(&HIPContext::mutex());
    });
}

namespace {

REGISTER_HIP_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<HIPContext>);

REGISTER_HIP_OPERATOR_WITH_ENGINE(
    CloneCommonWorld,
    GLOO,
    CloneCommonWorld<HIPContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
