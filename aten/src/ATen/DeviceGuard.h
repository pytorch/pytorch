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

  /// A constructor for testing; permits explicitly passing in the
  /// DeviceGuardImpl.
  explicit DeviceGuard(Device device, const detail::DeviceGuardImplInterface* impl) {
    init_device(device, impl);
  }

  /// Copy is disallowed.
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Note [Move construction for RAII guards is tricky]
  /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// In principle, move construction is useful for terminating
  /// the lifetime of a `DeviceGuard` early; for example:
  ///
  ///     // current device is d0
  ///     DeviceGuard g1(d1);
  ///     // current device is d1
  ///     {
  ///       DeviceGuard g2(std::move(g1));
  ///     }
  ///     // current device is d0!!
  ///
  /// However, it's difficult to implement the move constructor
  /// in a way that works in all situations.  For example, consider
  /// the following example:
  ///
  ///     DeviceGuard g1(d1);
  ///     {
  ///       DeviceGuard g2(d2);
  ///       {
  ///         DeviceGuard g3(std::move(g1)); // !!!
  ///       }
  ///     }
  ///
  /// What should the current device be while g3 in scope... and what
  /// should it be after it goes out of scope?  What about g2?
  /// There don't seem to be satisfactory answers for these questions.
  ///
  /// It's in principle possible to raise an error when this occurs
  /// by doing some extra thread-local bookkeeping.  But why bother?
  /// Just don't provide the constructor.
  DeviceGuard(DeviceGuard&& other) = delete;

  // Note [Move assignment for RAII guards is tricky]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Move assignment is deleted, because you need to know which guard was
  // defined "first", as that guard's original_device_ wins--with the current
  // representation, we have no way of telling which is the case.  (Move
  // construction does not have this problem, as one guard is always
  // uninitialized.)
  //
  // We can make this clear by way of a pair of examples:
  //
  // Example 1:
  //
  //  // initial device is n0
  //  {
  //    CUDAGuard g1(n1);
  //    {
  //      CUDAGuard g2(n2);
  //      // current device should be n2
  //      g1 = std::move(g2);
  //      // current device should still be n2
  //    }
  //    // current device should still be n2
  //  }
  //  // current device should be n0
  //
  //  Example 2 (flip the order of the two guards):
  //
  //  // initial device is n0
  //  {
  //    CUDAGuard g2(n2);
  //    {
  //      CUDAGuard g1(n1);
  //      // current device should be n1
  //      g1 = std::move(g2);
  //      // current device should be n2
  //    }
  //    // current device should be n0 (since g2 has been vacated)
  //  }
  //
  // In both examples, we need g1 to restore to n0 after move assignment.
  // However, in example 1, this is determined by the  restore value of g1
  // (prior to the move). In example 2, however, it is determined by the the
  // restore value of g2(!!). We don't know which one should win, without having
  // a way of telling which guard was allocated first.
  //
  // We could solve this with an extra thread-local variable.  But no one is
  // actually using move-assignment.  So just get rid of it.
  DeviceGuard& operator=(DeviceGuard&& other) = delete;

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
  void init_device(Device device, const detail::DeviceGuardImplInterface* impl = nullptr) {
    if (device.type() == at::kCPU) {
      return;
    }
    impl_ = impl ? impl : detail::getDeviceGuardImpl(device.type());
    if (impl) {
      AT_ASSERT(impl->type() == device.type());
    }
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
