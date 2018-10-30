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
/// The device is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_device` after construction, the
/// destructor will still reset the device to the one that was active at
/// construction time.
///
/// It's invalid to call `set_device` with devices from different device types;
/// a DeviceGuard only ever handles setting/resetting device for a single
/// device type.
struct DeviceGuard {
  /// Default constructor, does nothing.
  DeviceGuard() = default;

  // Note [Explicit initialization of optional fields]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Explicit initialization of original_device_ and last_device_ is
  // required to workaround an nvcc bug; see https://github.com/pytorch/pytorch/issues/12117

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device)
    : original_device_(), last_device_() { // See Note [Explicit initialization of optional fields]
    set_device(device);
  }

  explicit DeviceGuard(c10::optional<Device> device_opt)
    : original_device_(), last_device_() { // See Note [Explicit initialization of optional fields]
    if (device_opt.has_value()) {
      set_device(device_opt.value());
    }
  }

  /// Sets the current device to the device on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor)
    : original_device_(), last_device_() { // See Note [Explicit initialization of optional fields]
    set_device_from(tensor);
  }

  /// Sets the current device to the device on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors)
    : original_device_(), last_device_() { // See Note [Explicit initialization of optional fields]
    if (!tensors.empty()) {
      set_device_from(tensors.front());
    }
  }

  /// Copy is disallowed.
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move-constructs this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard(DeviceGuard&& other) noexcept
    : original_device_(), last_device_() { // See Note [Explicit initialization of optional fields]
    // Reuse move assignment implementation
    *this = std::move(other);
  }

  /// Move-assigns this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard& operator=(DeviceGuard&& other) noexcept {
    // We cannot use the default move assignment here.  Quoth the standard:
    //
    //    constexpr optional( optional&& other )
    //
    //    If other contains a value, then depending on whether *this contains a
    //    value, the contained value is either direct-initialized or assigned from
    //    *other (2) or std::move(*other) (3). Note that a moved-from optional
    //    still contains a value.
    //
    // Swapping works fine though.
    std::swap(this->original_device_, other.original_device_);
    std::swap(this->last_device_, other.last_device_);
    return *this;
  }

  /// Resets the device to the device that was active at construction of the
  /// guard.
  ~DeviceGuard() {
    // It should only not have a value if an index was never actually set.
    if (original_device_) {
      impl_->uncheckedSetDevice(*original_device_);
    }
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    // Fastpath for CPU.  Hopefully this can be inlined away in many cases.
    if (device.type() == at::kCPU) {
      return;
    }

    // Fastpath for the -1 device scenario.
    // TODO: I really hate that we can have -1 in Device. Ugh ugh ugh.
    if (device.index() == -1) {
      return;
    }
    AT_ASSERT(device.index() >= 0);

    // Retrieve the implementation
    if (!impl_) {
      impl_ = detail::getDeviceGuardImpl(device.type());
    } else {
      AT_ASSERTM(original_device_->type() == device.type(),
                 "DeviceGuard was originally used to change the device for ",
                 original_device_->type(), ", but set_device() was subsequently ",
                 "used to change the device for ", device.type(), ".  To change ",
                 "current device for a different device type, you must use a fresh "
                 "DeviceGuard.");
    }

    // Do the device switch
    if (original_device_) {
      impl_->setDevice(device);
    } else {
      original_device_ = impl_->exchangeDevice(device);
    }

    last_device_ = device;
  }

  /// Calls `set_device` with the `Tensor`'s current device, if it is not a
  /// CPU tensor. Does nothing if the `tensor` is not defined.
  void set_device_from(const Tensor& tensor) {
    if (tensor.defined()) {
      set_device(tensor.device());
    }
  }

  /// Returns the device that was set upon construction of the guard.
  optional<Device> original_device() const noexcept {
    return original_device_;
  }

  /// Returns the last device that was set via `set_device`, if any.
  optional<Device> last_device() const noexcept {
    return last_device_;
  }

 private:
  /// The original device that was active at construction of this object,
  /// for the device type that this DeviceGuard is changing.
  /// This is nullopt when you've allocated a DeviceGuard, but you haven't
  /// actually asked to switch devices: in this case, the "original"
  /// device is undetermined, because we haven't said which device type
  /// the original is for.
  optional<Device> original_device_;

  /// The last device that was set via `set_device`.  This is nullopt
  /// when you've allocated a DeviceGuard, but you haven't actually
  /// asked to switch devices.
  optional<Device> last_device_;

  // Cached pointer to the interface which actually implements the operations
  // needed for the DeviceGuard.
  const detail::DeviceGuardImplInterface* impl_ = nullptr;

  // Member invariants:
  //    !impl_ <==> !original_device_ <==> !last_device_
};
} // namespace at
