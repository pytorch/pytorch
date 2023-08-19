#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string>
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
          {"ipu", DeviceType::IPU},
          {"xpu", DeviceType::XPU},
          {"mkldnn", DeviceType::MKLDNN},
          {"opengl", DeviceType::OPENGL},
          {"opencl", DeviceType::OPENCL},
          {"ideep", DeviceType::IDEEP},
          {"hip", DeviceType::HIP},
          {"ve", DeviceType::VE},
          {"fpga", DeviceType::FPGA},
          {"ort", DeviceType::ORT},
          {"xla", DeviceType::XLA},
          {"lazy", DeviceType::Lazy},
          {"vulkan", DeviceType::Vulkan},
          {"mps", DeviceType::MPS},
          {"meta", DeviceType::Meta},
          {"hpu", DeviceType::HPU},
          {"mtia", DeviceType::MTIA},
          {"autort", DeviceType::AutoRT},
          {"privateuseone", DeviceType::PrivateUse1},
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
  if (device_string == get_privateuse1_backend()) {
    return DeviceType::PrivateUse1;
  }
  std::vector<const char*> device_names;
  for (const auto& it : types) {
    if (it.first) {
      device_names.push_back(it.first);
    }
  }
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", device_names),
      " device type at start of device string: ",
      device_string);
}
enum DeviceStringParsingState { START, INDEX_START, INDEX_REST, ERROR };

} // namespace

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  std::string device_name, device_index_str;
  DeviceStringParsingState pstate = DeviceStringParsingState::START;

  // The code below tries to match the string in the variable
  // device_string against the regular expression:
  // ([a-zA-Z_]+)(?::([1-9]\\d*|0))?
  for (size_t i = 0;
       pstate != DeviceStringParsingState::ERROR && i < device_string.size();
       ++i) {
    const char ch = device_string.at(i);
    switch (pstate) {
      case DeviceStringParsingState::START:
        if (ch != ':') {
          if (isalpha(ch) || ch == '_') {
            device_name.push_back(ch);
          } else {
            pstate = DeviceStringParsingState::ERROR;
          }
        } else {
          pstate = DeviceStringParsingState::INDEX_START;
        }
        break;

      case DeviceStringParsingState::INDEX_START:
        if (isdigit(ch)) {
          device_index_str.push_back(ch);
          pstate = DeviceStringParsingState::INDEX_REST;
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::INDEX_REST:
        if (device_index_str.at(0) == '0') {
          pstate = DeviceStringParsingState::ERROR;
          break;
        }
        if (isdigit(ch)) {
          device_index_str.push_back(ch);
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::ERROR:
        // Execution won't reach here.
        break;
    }
  }

  const bool has_error = device_name.empty() ||
      pstate == DeviceStringParsingState::ERROR ||
      (pstate == DeviceStringParsingState::INDEX_START &&
       device_index_str.empty());

  TORCH_CHECK(!has_error, "Invalid device string: '", device_string, "'");

  try {
    if (!device_index_str.empty()) {
      index_ = static_cast<c10::DeviceIndex>(c10::stoi(device_index_str));
    }
  } catch (const std::exception&) {
    TORCH_CHECK(
        false,
        "Could not parse device index '",
        device_index_str,
        "' in device string '",
        device_string,
        "'");
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
