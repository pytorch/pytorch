#include "caffe2/contrib/gloo/common_world_ops.h"

#include "caffe2/core/context_gpu.h"

#include <gloo/cuda.h>
#include <gloo/transport/tcp/device.h>

namespace caffe2 {
namespace gloo {

template <>
void CreateCommonWorld<CUDAContext>::initializeForContext() {
  static std::once_flag once;
  std::call_once(once, [&]() {
      // This is the first time we call Gloo code for a CUDAContext.
      // Share Caffe2 CUDA mutex with Gloo.
      ::gloo::CudaShared::setMutex(&CUDAContext::mutex());
    });
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<CUDAContext>);

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    CloneCommonWorld,
    GLOO,
    CloneCommonWorld<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
