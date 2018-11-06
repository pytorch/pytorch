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
 */
template <typename T>
class InlineDeviceGuard {
public:
  /// Default constructor, reads the current device so that
  /// we may reset the device to the current device on destruction.
  explicit InlineDeviceGuard()
    : original_device_(T().getDevice())
    , current_device_(original_device_)
    {}

  /// Set the current device to the passed Device.
  explicit InlineDeviceGuard(Device device)
    : original_device_(maybeExchangeDevice(device))
    , current_device_(maybeDeviceElse(device, original_device_))
    {}

  /// Set the current device to the passed Device
  explicit InlineDeviceGuard(optional<Device> device_opt)
    : original_device_(maybeExchangeDevice(device_opt))
    , current_device_(maybeDeviceElse(device_opt, original_device_))
    {}

  /// Set the current device index to the passed DeviceIndex.  (The
  /// device type is inferred from the template parameter T).
  explicit InlineDeviceGuard(DeviceIndex device_index)
    : InlineDeviceGuard(Device(T().type(), device_index)) {}


  /// Copy is disallowed
  InlineDeviceGuard(const InlineDeviceGuard<T>&) = delete;
  InlineDeviceGuard<T>& operator=(const InlineDeviceGuard<T>&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  InlineDeviceGuard(InlineDeviceGuard<T>&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  InlineDeviceGuard& operator=(InlineDeviceGuard<T>&& other) = delete;

  ~InlineDeviceGuard() {
    if (original_device_ == DeviceType::CPU) return;
    T().uncheckedSetDevice(original_device_);
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    auto index = device.index();
    if (index == -1) return;
    AT_ASSERT(index >= 0);
    T().setDevice(device);
    current_device_ = device;
  }

  /// Sets the device index to the given one.
  void set_index(DeviceIndex index) {
    set_device(Device(T().type(), index));
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() {
    return original_device_;
  }

  /// Returns the device that is currently set as the current device by this
  /// guard.  Note that this may not actually be the current device, if there
  /// was an intervening DeviceGuard.
  Device current_device() {
    return current_device_;
  }

private:
  Device original_device_;
  Device current_device_;

  // These helpers let us write the initializers more compactly.
  Device maybeExchangeDevice(optional<Device> d_opt) {
    if (d_opt.has_value() && d_opt->index() != -1) {
      return T().exchangeDevice(d_opt.value());
    } else {
      return T().getDevice();
    }
  }
  Device maybeDeviceElse(optional<Device> d_opt, Device d_def) {
    if (d_opt.has_value() && d_opt->index() != -1) {
      return d_opt.value();
    } else {
      return d_def;
    }
  }

};

}} // namespace c10::detail
