#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <memory>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using DeviceType = torch::headeronly::DeviceType;

// The torch::stable::Device class is a high-level C++ wrapper around
// the C shim Device APIs. We've modeled this class after c10::Device
// to provide a stable ABI-compatible device representation.
//
// This class provides:
// 1. A nicer UX much closer to c10::Device than the C APIs with DeviceHandle
// 2. Automatic memory management via shared_ptr
// 3. Compatibility with the stable ABI guarantees
class Device {
 private:
  std::shared_ptr<DeviceOpaque> dh_;

 public:
  // Construct a stable::Device from a DeviceType and optional device index
  // Default index is -1 (current device)
  /* implicit */ Device(DeviceType type, int32_t index = -1) {
    DeviceHandle ret;
    TORCH_ERROR_CODE_CHECK(
        torch_create_device(static_cast<int32_t>(type), index, &ret));
    dh_ = std::shared_ptr<DeviceOpaque>(ret, [](DeviceHandle dh) {
      TORCH_ERROR_CODE_CHECK(torch_delete_device(dh));
    });
  }

  // Construct a stable::Device from a string description
  // The string must follow the schema: (cpu|cuda|...)[:<device-index>]
  /* implicit */ Device(const std::string& device_string) {
    DeviceHandle ret;
    TORCH_ERROR_CODE_CHECK(
        torch_create_device_from_string(device_string.c_str(), &ret));
    dh_ = std::shared_ptr<DeviceOpaque>(ret, [](DeviceHandle dh) {
      TORCH_ERROR_CODE_CHECK(torch_delete_device(dh));
    });
  }

  // Construct a stable::Device from a DeviceHandle
  // Steals ownership from the DeviceHandle
  explicit Device(DeviceHandle dh)
      : dh_(dh, [](DeviceHandle dh) {
          TORCH_ERROR_CODE_CHECK(torch_delete_device(dh));
        }) {}

  // Copy and move constructors can be default since the underlying handle is a
  // shared_ptr
  Device(const Device& other) = default;
  Device(Device&& other) noexcept = default;

  // Copy and move assignment operators can be default since the underlying
  // handle is a shared_ptr
  Device& operator=(const Device& other) = default;
  Device& operator=(Device&& other) noexcept = default;

  // Destructor can be default: shared_ptr has custom deletion logic
  ~Device() = default;

  // Returns a borrowed reference to the DeviceHandle
  DeviceHandle get() const {
    return dh_.get();
  }

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

  void set_index(int32_t index) {
    TORCH_ERROR_CODE_CHECK(torch_device_set_index(dh_.get(), index));
  }

  // defined in device_inl.h to avoid circular dependencies
  DeviceType type() const;

  int32_t index() const {
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(torch_device_index(dh_.get(), &device_index));
    return device_index;
  }

  bool has_index() const {
    return index() != -1;
  }

  bool is_cuda() const {
    return type() == DeviceType::CUDA;
  }

  bool is_cpu() const {
    return type() == DeviceType::CPU;
  }

  // =============================================================================
  // END of C-shimified c10::Device APIs
  // =============================================================================
};

HIDDEN_NAMESPACE_END(torch, stable)
