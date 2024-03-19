#include "caffe2/contrib/gloo/broadcast_ops.h"

#include "caffe2/core/context_gpu.h"

#include <gloo/cuda_broadcast_one_to_all.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initializeAlgorithm() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::CudaBroadcastOneToAll<float>(
        init_.context, init_.template getOutputs<float>(), init_.size, root_));
  } else if (init_.template IsType<long>()) {
    algorithm_.reset(new ::gloo::CudaBroadcastOneToAll<long>(
        init_.context, init_.template getOutputs<long>(), init_.size, root_));
  } else if (init_.template IsType<int>()) {
    algorithm_.reset(new ::gloo::CudaBroadcastOneToAll<int>(
        init_.context, init_.template getOutputs<int>(), init_.size, root_));
  } else if (init_.template IsType<at::Half>()) {
    algorithm_.reset(new ::gloo::CudaBroadcastOneToAll<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size,
        root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
