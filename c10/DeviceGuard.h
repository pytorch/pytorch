#include <c10/detail/InlineDeviceGuard.h>

namespace c10 {

/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// The device is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_device` after construction, the
/// destructor will still reset the device to the one that was active at
/// construction time.
///
/// This device guard does NOT have an uninitialized state; it is guaranteed
/// to reset a device on exit.  If you are in a situation where you *might*
/// want to setup a guard (i.e., are looking for the moral equivalent
/// of optional<DeviceGuard>), see OptionalDeviceGuard.
class DeviceGuard {
public:
  /// No default constructor; see Note [Omitted default constructor from RAII]
  /// for why.
  explicit DeviceGuard() = delete;

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) : guard_(device) {}

  /// Copy is disallowed
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move is disallowed, as DeviceGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  DeviceGuard(DeviceGuard&& other) = delete;
  DeviceGuard& operator=(DeviceGuard&& other) = delete;

  /// Sets the device to the given one.  The specified device must be consistent
  /// with the device type originally specified during guard construction.
  ///
  /// TODO: The consistency check here is inconsistent with StreamGuard's
  /// behavior with set_stream, where a stream on a different device than
  /// the original one isn't an error; we just reset the stream and then
  /// switch devices.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the device index to the given one.  The device type is inferred
  /// from the original device type the guard was constructed with.
  void set_index(DeviceIndex index) {
    guard_.set_index(index);
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  Device current_device() const {
    return guard_.current_device();
  }

private:
  detail::InlineDeviceGuard<detail::VirtualGuardImpl> guard_;
};

/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// Unlike DeviceGuard, a OptionalDeviceGuard may be uninitialized.  This occurs
/// when you use the nullary constructor, or pass a nullopt to the constructor.
/// Uninitialized OptionalDeviceGuards do *nothing*; they do not know what the
/// original device was, and they do not reset on destruction.
///
/// An initialized OptionalDeviceGuard doesn't restore device to its value at
/// construction; it restores device to its value *at initialization*.  So if you
/// have the program:
///
///     setDevice(1);
///     OptionalDeviceGuard g;
///     setDevice(2);
///     g.set_device(3);
///
/// On destruction, g will reset device to 2, rather than 1.
///
/// An uninitialized OptionalDeviceGuard is distinct from a OptionalDeviceGuard whose
/// original_device and current_device match, since the OptionalDeviceGuard will
/// still reset the device to original_device.
class OptionalDeviceGuard {
public:
  /// Create an uninitialized guard.  Set the guard later using set_device.
  explicit OptionalDeviceGuard() : guard_() {}

  /// Initialize the guard, setting the current device to the passed Device.
  explicit OptionalDeviceGuard(Device device) : guard_(device) {}

  /// Initialize the guard if a Device is passed; otherwise leave the
  /// guard uninitialized.
  explicit OptionalDeviceGuard(optional<Device> device) : guard_(device) {}

  /// Copy is disallowed
  OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;

  /// Move is disallowed
  /// See Note [Explicit initialization of optional fields]
  /// and // Note [Move construction for RAII guards is tricky]
  /// for rationale.
  OptionalDeviceGuard(OptionalDeviceGuard&& other) = delete;
  OptionalDeviceGuard& operator=(OptionalDeviceGuard&& other) = delete;

  /// Sets the device to the given one.  The specified device must be consistent
  /// with the device type originally specified during guard construction.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// Returns the device that was set at the time the guard was constructed.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

private:
  detail::InlineOptionalDeviceGuard<detail::VirtualGuardImpl> guard_;
};

// Design note: in principle, we could avoid these wrappers using:
//
// using DeviceGuard = detail::InlineDeviceGuard<detail::VirtualGuardImpl>;
// using OptionalDeviceGuard = detail::InlineOptionalDeviceGuard<detail::VirtualGuardImpl>;
//
// But the error messages are worse, and our users can't just look at the
// header file to find out what's going on.  No, let's write out the API
// explicitly.

} // namespace c10
