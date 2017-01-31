#include "common_world_ops.h"

#include "fbcollective/transport/tcp/device.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(std::shared_ptr<::fbcollective::Context>);

namespace fbcollective {

template <>
std::shared_ptr<::fbcollective::transport::Device>
CreateCommonWorld<CPUContext>::createDevice() {
  // Share single device between all common worlds.
  // This should be made configurable, for varying transports, and
  // transport options (e.g. tcp socket options, ibverbs device).
  static auto sharedDevice = ::fbcollective::transport::tcp::CreateDevice();
  return sharedDevice;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    FBCOLLECTIVE,
    CreateCommonWorld<CPUContext>);

} // namespace fbcollective
} // namespace caffe2
