#include "common_world_ops.h"

#include "gloo/transport/tcp/device.h"

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
  ::gloo::transport::tcp::attr attr;
  static auto sharedDevice = ::gloo::transport::tcp::CreateDevice(attr);
  return sharedDevice;
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
