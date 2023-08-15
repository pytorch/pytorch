#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
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
using DeviceIndex = int8_t;

/// Represents a compute device on which a tensor is located. A device is
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
      : type_(type), index_(index) {
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

  /// Return true if the device is of PrivateUse1 type.
  bool is_privateuseone() const noexcept {
    return type_ == DeviceType::PrivateUse1;
  }

  /// Return true if the device is of MPS type.
  bool is_mps() const noexcept {
    return type_ == DeviceType::MPS;
  }

  /// Return true if the device is of HIP type.
  bool is_hip() const noexcept {
    return type_ == DeviceType::HIP;
  }

  /// Return true if the device is of VE type.
  bool is_ve() const noexcept {
    return type_ == DeviceType::VE;
  }

  /// Return true if the device is of XPU type.
  bool is_xpu() const noexcept {
    return type_ == DeviceType::XPU;
  }

  /// Return true if the device is of IPU type.
  bool is_ipu() const noexcept {
    return type_ == DeviceType::IPU;
  }

  /// Return true if the device is of XLA type.
  bool is_xla() const noexcept {
    return type_ == DeviceType::XLA;
  }

  /// Return true if the device is of MTIA type.
  bool is_mtia() const noexcept {
    return type_ == DeviceType::MTIA;
  }

  /// Return true if the device is of HPU type.
  bool is_hpu() const noexcept {
    return type_ == DeviceType::HPU;
  }

  /// Return true if the device is of Lazy type.
  bool is_lazy() const noexcept {
    return type_ == DeviceType::Lazy;
  }

  /// Return true if the device is of Vulkan type.
  bool is_vulkan() const noexcept {
    return type_ == DeviceType::Vulkan;
  }

  /// Return true if the device is of Metal type.
  bool is_metal() const noexcept {
    return type_ == DeviceType::Metal;
  }

  /// Return true if the device is of ORT type.
  bool is_ort() const noexcept {
    return type_ == DeviceType::ORT;
  }

  /// Return true if the device is of META type.
  bool is_meta() const noexcept {
    return type_ == DeviceType::Meta;
  }

  /// Return true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  /// Return true if the device supports arbitrary strides.
  bool supports_as_strided() const noexcept {
    return type_ != DeviceType::IPU && type_ != DeviceType::XLA &&
        type_ != DeviceType::Lazy && type_ != DeviceType::MTIA;
  }

  /// Same string as returned from operator<<.
  std::string str() const;

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;
  void validate() {
    // Removing these checks in release builds noticeably improves
    // performance in micro-benchmarks.
    // This is safe to do, because backends that use the DeviceIndex
    // have a later check when we actually try to switch to that device.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        index_ == -1 || index_ >= 0,
        "Device index must be -1 or non-negative, got ",
        (int)index_);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !is_cpu() || index_ <= 0,
        "CPU device index must be -1 or zero, got ",
        (int)index_);
  }
};

C10_API std::ostream& operator<<(std::ostream& stream, const Device& device);

} // namespace c10

namespace std {
template <>
struct hash<c10::Device> {
  size_t operator()(c10::Device d) const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(c10::DeviceType) == 1, "DeviceType is not 8-bit");
    static_assert(sizeof(c10::DeviceIndex) == 1, "DeviceIndex is not 8-bit");
    // Note [Hazard when concatenating signed integers]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We must first convert to a same-sized unsigned type, before promoting to
    // the result type, to prevent sign extension when any of the values is -1.
    // If sign extension occurs, you'll clobber all of the values in the MSB
    // half of the resulting integer.
    //
    // Technically, by C/C++ integer promotion rules, we only need one of the
    // uint32_t casts to the result type, but we put in both for explicitness's
    // sake.
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()))
            << 16 |
        static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
} // namespace std
