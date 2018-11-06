#pragma once

#include <c10/Device.h>
#include <c10/detail/DeviceGuardImplInterface.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace detail {

/**
 * InlineDeviceGuard is a helper class for implementing device type
 * specialized DeviceGuards, e.g., CUDAGuard.  The key idea is that
 * it is templated over DeviceGuardImpl, allowing it to devirtualize
 * all calls.  The intention is that InlineDeviceGuard<CUDAGuardImpl>
 * should be *as* efficient as straight line code that calls
 * cudaGetDevice-cudaSetDevice.  InlineDeviceGuard can only be used
 * from code that links against the relevant backend library, e.g.,
 * CUDA library.
 *
 * Users of InlineDeviceGuard may also find that the interface is also
 * more expressive.  This is because DeviceGuard cannot handle
 * set_device() in a reasonable way (consider what happens if you
 * set_device for CUDA, and then set_device for HIP!)  Because
 * InlineDeviceGuard is specialized for a specific device type,
 * it doesn't have to worry about this API situation.  (In principle,
 * you might find a use for this interface even without devirtualization;
 * but a better solution in such cases is to move your code into a compilation
 * unit that links against, e.g., CUDA).
 *
 * This code assumes that there are no non-bracketed calls to the underlying
 * setDevice() within the body of the guard.
 */
template <typename T>
class InlineDeviceGuard {
public:
  /// Default constructor, reads the current device so that
  /// we may reset the device to the current device on destruction.
  explicit InlineDeviceGuard() {
    initialize();
  }

  /// Set the current device to the passed Device.
  explicit InlineDeviceGuard(Device device) {
    set_device(device); // In Optimizer We Trust
    if (!initialized()) initialize();
  }

  /// Set the current device to the passed Device
  explicit InlineDeviceGuard(optional<Device> device_opt) {
    if (device_opt.has_value()) {
      set_device(device_opt.value());
      if (!initialized()) initialize();
    } else {
      initialize();
    }
  }

  /// Set the current device index to the passed DeviceIndex.  (The
  /// device type is inferred from the template parameter T).
  explicit InlineDeviceGuard(DeviceIndex device_index)
    : InlineDeviceGuard(Device(T().type(), device_index)) {}


  /// Copy is disallowed
  InlineDeviceGuard(const InlineDeviceGuard<T>&) = delete;
  InlineDeviceGuard<T>& operator=(const InlineDeviceGuard<T>&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  InlineDeviceGuard(InlineDeviceGuard<T>&& other) noexcept = delete;

  // See Note [Move assignment for RAII guards is tricky]
  InlineDeviceGuard& operator=(InlineDeviceGuard<T>&& other) noexcept = delete;

  ~InlineDeviceGuard() {
    if (original_device_ == DeviceType::CPU) return;
    T().uncheckedSetDevice(original_device_);
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    auto index = device.index();
    if (index == -1) return;
    AT_ASSERT(index >= 0);
    if (original_device_ == DeviceType::CPU) {
      original_device_ = T().exchangeDevice(device);
    } else {
      T().setDevice(device);
    }
    current_device_ = device;
  }

  /// Sets the device index to the given one.
  void set_index(DeviceIndex index) {
    set_device(Device(T().type(), index));
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() {
    AT_ASSERTM(original_device_ != DeviceType::CPU,
               "This device guard was moved-out from and is no longer valid");
    return original_device_;
  }

  /// Returns the device that is currently set as the current device by this
  /// guard.  Note that this may not actually be the current device, if there
  /// was an intervening DeviceGuard.
  Device current_device() {
    AT_ASSERTM(current_device_ != DeviceType::CPU,
               "This device guard was moved-out from and is no longer valid");
    return current_device_;
  }

  /// Returns whether or not the device guard is initialized.  A device guard
  /// becomes uninitialized when it is is moved out from.
  bool initialized() {
    return original_device_ != DeviceType::CPU;
  }

private:
  void initialize() {
    original_device_ = T().getDevice();
    current_device_ = original_device_;
  }

  // In principle, these could have been done alternately as optional<Device>,
  // but that runs afoul the lgenfe bug
  // (https://github.com/pytorch/pytorch/issues/12117) so I've instead
  // plundered CPU for the default value.
  //
  // The CPU state can only occur when you have moved out from a
  // device guard and so it is now "inert."  Note that a default
  // constructed device guard is not inert; it will reset to the
  // current device at construction time when it dies.
  Device original_device_ = DeviceType::CPU;
  Device current_device_ = DeviceType::CPU;
};

}} // namespace c10::detail
