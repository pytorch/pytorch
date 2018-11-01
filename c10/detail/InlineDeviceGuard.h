#pragma once

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
  /// Default constructor, does nothing.
  explicit InlineDeviceGuard() {}

  /// Set the current device to the passed Device.
  explicit InlineDeviceGuard(Device device) {
    set_device(device); // In Optimizer We Trust
  }

  /// Set the current device to the passed Device
  explicit InlineDeviceGuard(optional<Device> device_opt) {
    if (device_opt.has_value()) {
      set_device(device_opt.value());
    }
#ifdef DEBUG
    initialize_if_needed();
#endif
  }

  /// Set the current device index to the passed DeviceIndex.  (The
  /// device type is inferred from the template parameter T).
  explicit InlineDeviceGuard(DeviceIndex device_index)
    : InlineDeviceGuard(Device(T().type(), device_index)) {}


  /// Copy is disallowed
  InlineDeviceGuard(const InlineDeviceGuard<T>&) = delete;
  InlineDeviceGuard<T>& operator=(const InlineDeviceGuard<T>&) = delete;

  /// Move-constructs this `InlineDeviceGuard` from another `InlineDeviceGuard`. The
  /// moved-from `InlineDeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  InlineDeviceGuard(InlineDeviceGuard<T>&& other) noexcept {
    *this = std::move(other);
  }

  /// Move-assigns this `InlineDeviceGuard` from another `InlineDeviceGuard`. The
  /// moved-from `InlineDeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  InlineDeviceGuard& operator=(InlineDeviceGuard<T>&& other) noexcept {
    initialized_ = other.initialized_;
    original_device_ = other.original_device_;
    current_device_ = other.current_device_;
    other.initialized_ = false;
    return *this;
  } 

  ~InlineDeviceGuard() {
    debug_sanity_check();
    if (initialized_ && original_device_ != current_device_) {
      T().uncheckedSetDevice(original_device_);
    }
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    debug_sanity_check();
    auto index = device.index();
    if (index == -1) return;
    AT_ASSERT(index >= 0);
    if (initialized_) {
      if (current_device_ != device) {
        T().setDevice(device);
        current_device_ = device;
      }
    } else {
      initialized_ = true;
      original_device_ = T().exchangeDevice(device);
      current_device_ = device;
    }
  }

  /// Sets the device index to the given one.
  void set_index(DeviceIndex index) {
    set_device(Device(T().type(), index));
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() {
    initialize_if_needed();
    return original_device_;
  }

  /// Returns the device that is currently set as the current device by this
  /// guard.  Note that this may not actually be the current device, if there
  /// was an intervening DeviceGuard set.
  Device current_device() {
    initialize_if_needed();
    return current_device_;
  }

private:
  void initialize_if_needed() {
    debug_sanity_check();
    if (!initialized_) {
      original_device_ = T().getDevice();
      current_device_ = original_device_;
    }
  }

  void debug_sanity_check() {
#ifdef DEBUG
    // In DEBUG mode, InlineDeviceGuard is always initialized.  (This is not
    // true in non-DEBUG mode.)
    AT_ASSERT(initialized_);
    AT_ASSERT(T().getDevice() == current_device_);
#endif
  }

  bool initialized_ = false;
  Device original_device_ = at::kCPU;
  Device current_device_ = at::kCPU;

  // original_device_ and current_device_ are only valid if initialized == true
  // I could have put original_device_ and current_device_ in optional, but that
  // adds some illegal states representable (original_device_.has_value() !=
  // current_device_.has_value()) and would run afoul of
  // https://github.com/pytorch/pytorch/issues/12117 so on balance, it seemed
  // better to do it this way.
};

}} // namespace c10::detail
