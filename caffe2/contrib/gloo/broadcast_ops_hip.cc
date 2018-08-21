#include "broadcast_ops.h"

#include "caffe2/core/hip/context_hip.h"

#include <gloo/hip_broadcast_one_to_all.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initializeAlgorithm() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::HipBroadcastOneToAll<float>(
        init_.context, init_.template getOutputs<float>(), init_.size, root_));
  } else if (init_.template IsType<long>()) {
    algorithm_.reset(new ::gloo::HipBroadcastOneToAll<long>(
        init_.context, init_.template getOutputs<long>(), init_.size, root_));
  } else if (init_.template IsType<int>()) {
    algorithm_.reset(new ::gloo::HipBroadcastOneToAll<int>(
        init_.context, init_.template getOutputs<int>(), init_.size, root_));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::HipBroadcastOneToAll<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size,
        root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_HIP_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<HIPContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
