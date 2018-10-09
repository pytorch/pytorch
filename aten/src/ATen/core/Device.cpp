#include <ATen/core/Device.h>
#include <ATen/core/Error.h>
#include <ATen/core/Macros.h>

#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace at {
namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<std::string, DeviceType>, 7> types = {{
      {"cpu", DeviceType::CPU},
      {"cuda", DeviceType::CUDA},
      {"mkldnn", DeviceType::MKLDNN},
      {"opengl", DeviceType::OPENGL},
      {"opencl", DeviceType::OPENCL},
      {"ideep", DeviceType::IDEEP},
      {"hip", DeviceType::HIP},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [device_string](const std::pair<std::string, DeviceType>& p) {
        return p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  AT_ERROR(
      "Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, or hip device type at start of device string");
}
} // namespace

// `std::regex` is still in a very incomplete state in GCC 4.8.x,
// so we have to do our own parsing, like peasants.
// https://stackoverflow.com/questions/12530406/is-gcc-4-8-or-earlier-buggy-about-regular-expressions
//
// Replace with the following code once we shed our GCC skin:
//
// static const std::regex regex(
//     "(cuda|cpu)|(cuda|cpu):([0-9]+)|([0-9]+)",
//     std::regex_constants::basic);
// std::smatch match;
// const bool ok = std::regex_match(device_string, match, regex);
// AT_CHECK(ok, "Invalid device string: '", device_string, "'");
// if (match[1].matched) {
//   type_ = parse_type_from_string(match[1].str());
// } else {
//   if (match[2].matched) {
//     type_ = parse_type_from_string(match[1].str());
//   } else {
//     type_ = Type::CUDA;
//   }
//   AT_ASSERT(match[3].matched);
//   index_ = std::stoi(match[3].str());
// }
Device::Device(const std::string& device_string) : Device(Type::CPU) {
  AT_CHECK(!device_string.empty(), "Device string must not be empty");
  int index = device_string.find(":");
  if (index == std::string::npos) {
    type_ = parse_type(device_string);
    return;
  } else {
    std::string s;
    s = device_string.substr(0, index);
    AT_CHECK(!s.empty(), "Device string must not be empty");
    type_ = parse_type(s);
  }
  std::string device_index = device_string.substr(index + 1);
  try {
    index_ = at::stoi(device_index);
  } catch (const std::exception&) {
    AT_ERROR(
        "Could not parse device index '",
        device_index,
        "' in device string '",
        device_string,
        "'");
  }
}

std::ostream& operator<<(std::ostream& stream, const at::Device& device) {
  stream << device.type();
  if (device.has_index()) {
    stream << ":" << device.index();
  }
  return stream;
}

} // namespace at
