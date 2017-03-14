#include "broadcast_ops.h"

#include "caffe2/core/context_gpu.h"

#include "gloo/cuda_broadcast_one_to_all.h"

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initialize() {
  auto* output = Output(OUTPUT);
  const auto& context =
      OperatorBase::Input<std::shared_ptr<::gloo::Context>>(COMM);
  if (output->template IsType<float>()) {
    auto ptrs = getPointers<float>();
    algorithm_.reset(new ::gloo::CudaBroadcastOneToAll<float>(
        context, ptrs, output->size(), root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", output->meta().name());
  }
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
