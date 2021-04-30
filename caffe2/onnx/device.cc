#include "caffe2/onnx/device.h"

#include <cstdlib>
#include <unordered_map>

namespace caffe2 { namespace onnx {
static const std::unordered_map<std::string, DeviceType> kDeviceMap = {
  {"CPU", DeviceType::CPU},
  {"CUDA", DeviceType::CUDA}
};

Device::Device(const std::string &spec) {
  auto pos = spec.find_first_of(':');
  type = kDeviceMap.at(spec.substr(0, pos - 1));
  device_id = atoi(spec.substr(pos + 1).c_str());
}
}}
