#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <torch/csrc/stable/accelerator.h>

#include <cctype>
#include <string>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using DeviceType = torch::headeronly::DeviceType;
using DeviceIndex = torch::stable::accelerator::DeviceIndex;

namespace {

DeviceType parse_device_type(const std::string& device_name) {
  if (device_name == "cpu")
    return DeviceType::CPU;
  if (device_name == "cuda")
    return DeviceType::CUDA;
  if (device_name == "xpu")
    return DeviceType::XPU;
  if (device_name == "mkldnn")
    return DeviceType::MKLDNN;
  if (device_name == "opengl")
    return DeviceType::OPENGL;
  if (device_name == "opencl")
    return DeviceType::OPENCL;
  if (device_name == "ideep")
    return DeviceType::IDEEP;
  if (device_name == "hip")
    return DeviceType::HIP;
  if (device_name == "ve")
    return DeviceType::VE;
  if (device_name == "fpga")
    return DeviceType::FPGA;
  if (device_name == "maia")
    return DeviceType::MAIA;
  if (device_name == "xla")
    return DeviceType::XLA;
  if (device_name == "lazy")
    return DeviceType::Lazy;
  if (device_name == "vulkan")
    return DeviceType::Vulkan;
  if (device_name == "mps")
    return DeviceType::MPS;
  if (device_name == "meta")
    return DeviceType::Meta;
  if (device_name == "hpu")
    return DeviceType::HPU;
  if (device_name == "mtia")
    return DeviceType::MTIA;
  if (device_name == "ipu")
    return DeviceType::IPU;
  if (device_name == "privateuseone")
    return DeviceType::PrivateUse1;

  STD_TORCH_CHECK(
      false,
      "Invalid device type: '",
      device_name,
      "'. Expected one of: cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, ipu, privateuseone");
}

} // namespace

// The torch::stable::Device class is an approximate copy of c10::Device.
// It has some slight modifications e.g. TORCH_INTERNAL_ASSERT_DEBUG_ONLY ->
// STD_TORCH_CHECK We chose to copy it rather than moving it to headeronly as
// (1) Device is < 8 bytes so the *Handle approach used for tensor doesn't make
// sense (2) c10::Device is not header-only due to its string constructor
// StableIValue conversions handle conversion between c10::Device (in libtorch)
// and torch::stable::Device (in stable user extensions)

class Device {
 private:
  DeviceType type_;
  DeviceIndex index_ = -1;

  void validate() {
    STD_TORCH_CHECK(
        index_ >= -1,
        "Device index must be -1 or non-negative, got ",
        static_cast<int>(index_));
    STD_TORCH_CHECK(
        type_ != DeviceType::CPU || index_ <= 0,
        "CPU device index must be -1 or zero, got ",
        static_cast<int>(index_));
  }

 public:
  // Construct a stable::Device from a DeviceType and optional device index
  // Default index is -1 (current device)
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    validate();
  }

  // Construct a stable::Device from a string description
  // The string must follow the schema: (cpu|cuda|...)[:<device-index>]
  /* implicit */ Device(const std::string& device_string)
      : Device(DeviceType::CPU) {
    STD_TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

    std::string device_name, device_index_str;
    enum ParsingState { START, INDEX_START, INDEX_REST, ERROR };
    ParsingState pstate = START;

    // Match: ([a-zA-Z_]+)(?::([1-9]\\d*|0))?
    for (size_t i = 0; pstate != ERROR && i < device_string.size(); ++i) {
      const char ch = device_string.at(i);
      const unsigned char uch = static_cast<unsigned char>(ch);
      switch (pstate) {
        case START:
          if (ch != ':') {
            if (std::isalpha(uch) || ch == '_') {
              device_name.push_back(ch);
            } else {
              pstate = ERROR;
            }
          } else {
            pstate = INDEX_START;
          }
          break;
        case INDEX_START:
          if (std::isdigit(uch)) {
            device_index_str.push_back(ch);
            pstate = INDEX_REST;
          } else {
            pstate = ERROR;
          }
          break;
        case INDEX_REST:
          if (device_index_str.at(0) == '0') {
            pstate = ERROR;
            break;
          }
          if (std::isdigit(uch)) {
            device_index_str.push_back(ch);
          } else {
            pstate = ERROR;
          }
          break;
        case ERROR:
          break;
      }
    }

    const bool has_error = device_name.empty() || pstate == ERROR ||
        (pstate == INDEX_START && device_index_str.empty());

    STD_TORCH_CHECK(!has_error, "Invalid device string: '", device_string, "'");

    try {
      if (!device_index_str.empty()) {
        index_ = static_cast<DeviceIndex>(std::stoi(device_index_str));
      }
    } catch (const std::exception&) {
      STD_TORCH_CHECK(
          false,
          "Could not parse device index '",
          device_index_str,
          "' in device string '",
          device_string,
          "'");
    }

    // Parse device type
    type_ = parse_device_type(device_name);
    validate();
  }

  // Copy and move constructors can be default
  Device(const Device& other) = default;
  Device(Device&& other) noexcept = default;

  // Copy and move assignment operators can be default
  Device& operator=(const Device& other) = default;
  Device& operator=(Device&& other) noexcept = default;

  // Destructor can be default
  ~Device() = default;

  // =============================================================================
  // C-shimified c10::Device APIs: the below APIs have the same signatures and
  // semantics as their counterparts in c10/core/Device.h.
  // =============================================================================

  bool operator==(const Device& other) const noexcept {
    return type() == other.type() && index() == other.index();
  }

  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  void set_index(DeviceIndex index) {
    index_ = index;
  }

  DeviceType type() const noexcept {
    return type_;
  }

  DeviceIndex index() const noexcept {
    return index_;
  }

  bool has_index() const noexcept {
    return index_ != -1;
  }

  bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  // =============================================================================
  // END of C-shimified c10::Device APIs
  // =============================================================================
};

HIDDEN_NAMESPACE_END(torch, stable)
