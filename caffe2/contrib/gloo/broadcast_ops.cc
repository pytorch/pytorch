#include "broadcast_ops.h"

#include <gloo/broadcast_one_to_all.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initializeAlgorithm() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<float>(
        init_.context, init_.template getOutputs<float>(), init_.size, root_));
  } else if (init_.template IsType<long>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<long>(
        init_.context, init_.template getOutputs<long>(), init_.size, root_));
  } else if (init_.template IsType<int>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<int>(
        init_.context, init_.template getOutputs<int>(), init_.size, root_));
  } else if (init_.template IsType<at::Half>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size,
        root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
