#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace c10 {
namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<
      std::pair<const char*, DeviceType>,
      static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"xpu", DeviceType::XPU},
          {"mkldnn", DeviceType::MKLDNN},
          {"opengl", DeviceType::OPENGL},
          {"opencl", DeviceType::OPENCL},
          {"ideep", DeviceType::IDEEP},
          {"hip", DeviceType::HIP},
          {"ve", DeviceType::VE},
          {"fpga", DeviceType::FPGA},
          {"msnpu", DeviceType::MSNPU},
          {"xla", DeviceType::XLA},
          {"lazy", DeviceType::Lazy},
          {"vulkan", DeviceType::Vulkan},
          {"mlc", DeviceType::MLC},
          {"meta", DeviceType::Meta},
          {"hpu", DeviceType::HPU},
      }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&device_string](const std::pair<const char*, DeviceType>& p) {
        return p.first && p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  TORCH_CHECK(
      false,
      "Expected one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, msnpu, mlc, xla, lazy, vulkan, meta, hpu device type at start of device string: ",
      device_string);
}
} // namespace

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  const size_t colon_pos = device_string.find(':');
  const bool has_device_index = colon_pos != std::string::npos;
  std::string device_name;

  if (has_device_index) {
    const size_t colon_pos_last = device_string.find_last_of(':');
    TORCH_CHECK(
        colon_pos == colon_pos_last,
        "Invalid device string: '",
        device_string,
        "'");
    device_name = device_string.substr(0, colon_pos);
    const std::string device_index_str = device_string.substr(colon_pos + 1);
    try {
      size_t next_valid_pos = 0;
      index_ = c10::stoi(device_index_str, &next_valid_pos);

      // Ensure that the entire string was consumed.
      TORCH_CHECK(
          next_valid_pos == device_index_str.size(),
          "Invalid device string: '",
          device_string,
          "' which has additional characters after the device index");
    } catch (const std::exception&) {
      TORCH_CHECK(
          false,
          "Invalid device string: '",
          device_string,
          "' which has an invalid device index");
    }
    TORCH_CHECK(
        index_ >= 0,
        "Invalid device string: '",
        device_string,
        "' which has a negative device index");
  } else {
    device_name = device_string;
  }
  type_ = parse_type(device_name);
  validate();
}

std::string Device::str() const {
  std::string str = DeviceTypeName(type(), /* lower case */ true);
  if (has_index()) {
    str.push_back(':');
    str.append(to_string(index()));
  }
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}

} // namespace c10
