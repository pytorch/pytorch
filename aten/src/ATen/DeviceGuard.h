#pragma once

#include <ATen/Tensor.h>
#include <c10/Device.h>
#include <ATen/core/ScalarType.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include "c10/util/Optional.h"
#include <c10/detail/DeviceGuardImplInterface.h>

#include <cstddef>

namespace at {
/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device (for that device type) that was originally
/// active upon destruction.
///
/// If a DeviceGuard is constructed without specifying a device type (this can
/// occur if you, e.g., pass a nullopt to the constructor), it behaves as if it
/// were a no-op "CPU" guard; e.g., current_device() reports that the current
/// device is kCPU.  This is different from passing Device(kCUDA, -1), which
/// says to use the current CUDA device; in this case, we will correctly query
/// what the current CUDA device is, won't change it, but WILL reset it
/// at the end of DeviceGuard.
class DeviceGuard {
public:
  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) {
    init_device(device);
  }

  /// Set the current device to the passed Device, if not nullopt;
  /// otherwise do nothing.
  explicit DeviceGuard(optional<Device> device_opt) {
    if (device_opt.has_value()) {
      init_device(device_opt.value());
    }
  }

  /// Sets the current device to the device on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor) {
    init_device_from(tensor);
  }

  /// Sets the current device to the device on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors) {
    if (!tensors.empty()) {
      init_device_from(tensors.front());
    }
  }

  /// Copy is disallowed.
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move-constructs this `DeviceGuard` from another `DeviceGuard`.
  /// This can be used to terminate the lifetime of a `DeviceGuard`
  /// early; for example, in:
  ///
  ///     // current device is d0
  ///     DeviceGuard g1(d1);
  ///     // current device is d1
  ///     {
  ///       DeviceGuard g2(std::move(g1));
  ///     }
  ///     // current device is d0!!
  ///
  /// It is undefined behavior if you move a device guard across
  /// an intervening DeviceGuard.
  ///
  ///     DeviceGuard g1(d1);
  ///     DeviceGuard g2(d2);
  ///     DeviceGuard g3(std::move(g1)); // UB!
  ///
  DeviceGuard(DeviceGuard&& other) noexcept {
    // Default construction leaves this uninitialized
    std::swap(impl_, other.impl_);
    std::swap(original_device_, other.original_device_);
    std::swap(current_device_, other.current_device_);
  }

  /// Move-assigns this `DeviceGuard` from another `DeviceGuard`. This
  /// `DeviceGuard` is immediately terminated.  This allows
  /// you to implement a modest performance optimization: if the device
  /// type of the previous DeviceGuard and the new DeviceGuard match,
  /// then skips the otherwise unnecessary setDevice from the previous
  /// device guard.
  ///
  /// As with the move constructor, it is undefined behavior if you
  /// move a device guard across an intervening DeviceGuard.
  ///
  DeviceGuard& operator=(DeviceGuard&& other) noexcept {
    if (other.original_device_.type() == original_device_.type()) {
      // other has already set the device to the desired new value;
      // cancel its destruction and update current_device.  Don't
      // update original_device, since we are still obligated
      // to restore to it at the very end.
      current_device_ = other.current_device_;
    } else {
      // the devices are unrelated, so just terminate the
      // current guard and then move other in
      if (original_device_ != current_device_) {
        AT_ASSERT(impl_); // see member invariant
        impl_->setDevice(original_device_);
      }
      impl_ = other.impl_;
      original_device_ = other.original_device_;
      current_device_ = other.current_device_;
    }
    other.impl_ = nullptr;
    other.current_device_ = at::kCPU;
    other.original_device_ = at::kCPU;
    return *this;
  }

  /// Resets the device to the device that was active at construction of the
  /// guard.
  ~DeviceGuard() {
    // This optimization is sound if we are guaranteed not to have
    // any unmanaged setDevice calls inside the body of the guard.
    // if (original_device_ == current_device_) return;
    if (!impl_) return;
    impl_->uncheckedSetDevice(original_device_);
  }

  /// Returns the device that was set prior to construction of the guard.
  Device original_device() const noexcept {
    return original_device_;
  }

  /// Returns the device that was set after construction of the guard.
  Device current_device() const noexcept {
    return current_device_;
  }

 private:
  void init_device(Device device) {
    if (device.type() == at::kCPU) {
      return;
    }
    impl_ = detail::getDeviceGuardImpl(device.type());
    if (device.index() == -1) {
      original_device_ = impl_->getDevice();
      current_device_ = original_device_;
    } else {
      original_device_ = impl_->exchangeDevice(device);
      current_device_ = device;
    }
  }

  void init_device_from(const Tensor& tensor) {
    if (tensor.defined()) {
      init_device(tensor.device());
    }
  }

  /// The original device that was active at construction of this object,
  /// for the device type that this DeviceGuard is changing.  Defaults to
  /// kCPU if no device type is specified for DeviceGuard.
  Device original_device_ = at::kCPU;

  /// The last device that was set via `set_device`, or the previous
  /// device, if no device was set.  Defaults to kCPU if no device type is
  /// specified for DeviceGuard.
  Device current_device_ = at::kCPU;

  /// Cached pointer to the interface which actually implements the operations
  /// needed for the DeviceGuard.  This is nullptr if the guard is for CPU.
  const detail::DeviceGuardImplInterface* impl_ = nullptr;

  // Member invariants:
  //    !impl_ <==> original_device_ == at::kCPU <==> current_device_ == at::kCPU
};
} // namespace at
