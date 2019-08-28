#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vectorgex(
//     "(cuda|cpu)|(cuda|cpu):([0-9]+)|([0-9]+)",
//     std::regex_constants::basic);
// std::smatch match;
// const bool ok = std::regex_match(device_string, match, regex);
// TORCH_CHECK(ok, "Invalid device string: '", device_string, "'");
// if (match[1].matched) {
//   type_ = parse_type_from_string(maype::MKLDNN},
      {"opengl", DeviceType::OPENGL},
      {"opencl", DeviceType::OPENCL},
      {"ideep", DeviceType::IDEEP},
      {"hip", DeviceType::HIP},
      {"msnpu", DeviceType::MSNPU},
      {"xla", DeviceTygex(
//     "(cuda|cpu)|(cuda|cpu):([0-9]+)|([0-9]+)",
//     std::regex_constants::basic);
// std::smatch match;
// const bool ok = std::regex_match(device_string, match, regex);
// TORCH_CHECK(ok, "Invalid device string: '", device_string, "'");
// if (match[1].matched) {
//   type_ = parse_type_from_string(man a very incomplete state in GCC 4.8.x,
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
// TORCH_CHECK(ok, "Invalid device string: '", device_string, "'");
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
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");
  int index = device_string.find(":");
  if (index == std::string::npos) {
    type_ = parse_type(device_string);
  } else {
    std::string s;
    s = device_stgex(
//     "(cuda|cpu)|(cuda|cpu):([0-9]+)|([0-9]+)",
//     std::regex_constants::basic);
// std::smatch match;
// const bool ok = std::regex_match(device_string, match, regex);
// TORCH_CHECK(ok, "Invalid device string: '", device_string, "'");
// if (match[1].matched) {
//   type_ = parse_type_from_string(mat parse device index '", device_index,
               "' in device string '", device_string, "'");
    }
    TORCH_CHECK(index_ >= 0,
             "Device index must be non-negative, got ", index_);
  }
  validate();
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.type();
  if (device.has_index()) {
    stream << ":" << device.index();
  }
  return stream;
}

} // namespace c10
