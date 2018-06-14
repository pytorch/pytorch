#include <ATen/Device.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <exception>
#include <ostream>
#include <string>
#include <tuple>

namespace at {
namespace {
std::pair<optional<Device::Type>, size_t> parse_type(
    const std::string& device_string) {
  auto position = device_string.find("cpu");
  if (position != std::string::npos) {
    return {Device::Type::CPU, 3};
  }
  position = device_string.find("cuda");
  if (position != std::string::npos) {
    return {Device::Type::CUDA, 4};
  }
  return {nullopt, 0};
}

optional<int32_t> device_index_if_cuda_tensor(Tensor tensor) {
  if (tensor.is_cuda()) {
    return tensor.get_device();
  }
  return nullopt;
}
} // namespace

Device::Device(Tensor tensor)
    : Device(tensor.type().backend(), device_index_if_cuda_tensor(tensor)) {}

Device::Device(Type type, at::optional<int32_t> index)
    : type_(type), index_(index) {
  AT_CHECK(index.value_or(0) >= 0, "Device index must not be negative");
  AT_CHECK(
      !is_cpu() || index.value_or(0) == 0, "CPU device index must be zero");
}

Device::Device(Type type) : Device(type, nullopt) {}

Device::Device(Backend backend, at::optional<int32_t> index)
    : Device(backend_to_type(backend), index) {}

Device::Device(Backend backend) : Device(backend, nullopt) {}

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
  at::optional<Device::Type> optional_type;
  size_t position;
  std::tie(optional_type, position) = parse_type(device_string);
  if (optional_type.has_value()) {
    type_ = optional_type.value();
  }

  // e.g. 'cuda', 'cpu' or the empty string.
  if (position == device_string.size()) {
    return;
  }

  // e.g. 'cpu:1', 'cuda:123'
  if (device_string[position] == ':') {
    AT_CHECK(
        optional_type.has_value(),
        "Found ':' in device string '",
        device_string,
        "', but neither 'cpu' nor 'cuda' preceded it");
    // Skip the colon.
    position += 1;
  }

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

  // Now that we have a device index, the device type defaults to CUDA.
  // e.g. '1', '123'
  type_ = optional_type.value_or(Type::CUDA);
}

void Device::set_index(at::optional<int32_t> index) {
  index_ = index;
}

} // namespace at

std::ostream& operator<<(std::ostream& stream, at::Device::Type type) {
  switch (type) {
    case at::Device::Type::CPU: {
      stream << "cpu";
      break;
    }
    case at::Device::Type::CUDA: {
      stream << "cuda";
      break;
    }
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const at::Device& device) {
  stream << device.type();
  if (device.has_index()) {
    stream << ":" << device.index().value();
  }
  return stream;
}
