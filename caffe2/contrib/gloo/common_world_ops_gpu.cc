#include "common_world_ops.h"

#include "caffe2/core/context_gpu.h"

#include <gloo/cuda.h>
#include <gloo/transport/tcp/device.h>

namespace caffe2 {
namespace gloo {

template <typename T>
std::shared_ptr<::gloo::transport::Device>
CreateCommonWorld<T>::createDevice() {
  // Share single device between all common worlds. This should be
  // made configurable, for varying transports, and transport options
  // (e.g. tcp socket options, ibverbs device).
  //
  // All pairs are switched to synchronous mode after having
  // connected, so they don't need to synchronize with the device
  // thread when they are used from an algorithm.
  //
  static std::once_flag once;
  static std::shared_ptr<::gloo::transport::Device> device;
  std::call_once(once, [&]() {
    ::gloo::transport::tcp::attr attr;
    device = ::gloo::transport::tcp::CreateDevice(attr);

    // This operator is the first time any Gloo code is executed
    // for a CUDAContext. Share Caffe2 CUDA mutex with Gloo.
    ::gloo::CudaShared::setMutex(&CUDAContext::mutex());
  });

  return device;
}

namespace {

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<CUDAContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
