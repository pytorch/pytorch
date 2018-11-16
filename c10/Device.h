#pragma once

#include <c10/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>

namespace c10 {

/// An index representing a specific device; e.g., the 1 in GPU 1.
/// A DeviceIndex is not independently meaningful without knowing
/// the DeviceType it is associated; try to use Device rather than
/// DeviceIndex directly.
using DeviceIndex = int16_t;

/// Represents a a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct C10_API Device {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    AT_CHECK(
        index == -1 || index >= 0,
        "Device index must be -1 or non-negative, got ",
        index);
    AT_CHECK(
        !is_cpu() || index <= 0,
        "CPU device index must be -1 or zero, got ",
        index);
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda):[<device-index>]`
  /// where `cpu:` or `cuda:` specifies the device type, and
  /// `<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  /// Returns true if the type and index of this `Device` matches that of
  /// `other`.
  bool operator==(const Device& other) const noexcept {
    return this->type_ == other.type_ && this->index_ == other.index_;
  }

  /// Returns true if the type or index of this `Device` differs from that of
  /// `other`.
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  /// Sets the device index.
  void set_index(DeviceIndex index) {
    index_ = index;
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns the optional index.
  DeviceIndex index() const noexcept {
    return index_;
  }

  /// Returns true if the device has a non-default index.
  bool has_index() const noexcept {
    return index_ != -1;
  }

  /// Return true if the device is of CUDA type.
  bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  /// Return true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;
};

C10_API std::ostream& operator<<(
    std::ostream& stream,
    const Device& device);

} // namespace c10

namespace std {
template <>
struct hash<c10::Device> {
  size_t operator()(c10::Device device) const noexcept {
    size_t hash_val = static_cast<size_t>(device.index() + 1);
    if (device.is_cuda()) {
      hash_val += 2;
    }
    return hash_val;
  }
};
} // namespace std

// TODO: Remove when we add global namespace include
namespace at {
using c10::Device;
using c10::DeviceIndex;
}
