#include <ATen/Device.h>

#include <ATen/core/Error.h>

#include <exception>
#include <ostream>
#include <string>
#include <tuple>

namespace at {
namespace {
std::pair<Device::Type, size_t> parse_type(const std::string& device_string) {
  auto position = device_string.find("cpu");
  if (position != std::string::npos) {
    return {Device::Type::CPU, 3};
  }
  position = device_string.find("cuda");
  if (position != std::string::npos) {
    return {Device::Type::CUDA, 4};
  }
  AT_ERROR("Expected 'cpu' or 'cuda' device type at start of device string");
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

  size_t position;
  std::tie(type_, position) = parse_type(device_string);

  // e.g. 'cuda', 'cpu'.
  if (position == device_string.size()) {
    return;
  }

  AT_CHECK(
      device_string[position] == ':',
      "Expected ':' to separate device type from index in device string");
  // Skip the colon.
  position += 1;

  const auto index_string = device_string.substr(position);
  try {
    index_ = std::stoi(index_string);
  } catch (const std::exception&) {
    AT_ERROR(
        "Could not parse device index '",
        index_string,
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
