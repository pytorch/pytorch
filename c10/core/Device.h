#pragma once

#include <c10/core/DeviceType.h>
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

namespace detail {

  // Are you here because this static assert failed?  Make sure you ensure
  // that the bitmasking code below is updated accordingly!
  static_assert(sizeof(c10::DeviceType) == 2, "DeviceType is not 16-bit");
  static_assert(sizeof(c10::DeviceIndex) == 2, "DeviceIndex is not 16-bit");

  inline uint32_t packDevice(DeviceType type_enum, DeviceIndex index) {
    int16_t type = static_cast<int16_t>(type_enum);
    uint16_t type_bit;
    memcpy(&type_bit, &type, sizeof(uint16_t));
    uint16_t index_bit;
    memcpy(&index_bit, &index, sizeof(uint16_t));
    return (static_cast<uint32_t>(type_bit) << 16) | index_bit;
  }

  inline std::pair<DeviceType, DeviceIndex> unpackDevice(uint32_t pack) {
    uint16_t type_bit = pack >> 16;
    uint16_t index_bit = pack & 0xFFFF;
    int16_t type;
    memcpy(&type, &type_bit, sizeof(int16_t));
    int16_t index;
    memcpy(&index, &index_bit, sizeof(int16_t));
    return std::make_pair(static_cast<DeviceType>(type), index);
  }

}

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
struct C10_API Device final {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : data_(detail::packDevice(type, index)) {
    validate();
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  /// Returns true if the type and index of this `Device` matches that of
  /// `other`.
  bool operator==(const Device& other) const noexcept {
    return this->data_ == other.data_;
  }

  /// Returns true if the type or index of this `Device` differs from that of
  /// `other`.
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  /// Sets the device index.
  void set_index(DeviceIndex index) {
    auto p = detail::unpackDevice(data_);
    data_ = detail::packDevice(p.first, index);
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return detail::unpackDevice(data_).first;
  }

  /// Returns the optional index.
  DeviceIndex index() const noexcept {
    return detail::unpackDevice(data_).second;
  }

  /// Returns true if the device has a non-default index.
  bool has_index() const noexcept {
    return index() != -1;
  }

  /// Return true if the device is of CUDA type.
  bool is_cuda() const noexcept {
    return type() == DeviceType::CUDA;
  }

  /// Return true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type() == DeviceType::CPU;
  }

  /// Same string as returned from operator<<.
  std::string str() const;

 private:
  // Two adjacent int16_t fields has field access miscompiled on NVCC.
  // To workaround this problem, we do the packing and unpacking
  // manually.  FB employees can see
  //   https://fb.workplace.com/groups/llvm.gcc/permalink/4053565044692080/
  // for more details
  uint32_t data_;
  void validate() {
    TORCH_CHECK(index_ == -1 || index_ >= 0,
        "Device index must be -1 or non-negative, got ", index_);
    TORCH_CHECK(!is_cpu() || index_ <= 0,
        "CPU device index must be -1 or zero, got ", index_);
  }
};

C10_API std::ostream& operator<<(
    std::ostream& stream,
    const Device& device);

} // namespace c10

namespace std {
template <>
struct hash<c10::Device> {
  size_t operator()(c10::Device d) const noexcept {
    return std::hash<uint32_t>{}(detail::packDevice(d.type(), d.index()));
  }
};
} // namespace std
