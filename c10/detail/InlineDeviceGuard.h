#pragma once

#include <c10/Device.h>
#include <c10/detail/DeviceGuardImplInterface.h>
#include <c10/detail/VirtualGuardImpl.h>
#include <c10/util/Optional.h>
#include <c10/util/C++17.h>

namespace c10 {
namespace detail {

/**
 * A DeviceGuard is an RAII class that sets a device to some value
 * on construction, and resets the device to its original value on
 * destruction.
 *
 * InlineDeviceGuard is a helper class for implementing DeviceGuards.
 * The key idea is that it is templated over DeviceGuardImpl, which
 * means that if a concrete implementation is provided, we can
 * devirtualize all calls.  The intention is that
 * InlineDeviceGuard<CUDAGuardImpl> should be *as* efficient as straight line
 * code that calls cudaGetDevice-cudaSetDevice.  Additionally, this
 * class is written in a way that a virtualized implementation is
 * possible via VirtualGuardImpl.
 *
 * InlineDeviceGuard is always initialized, and always resets device on exit.
 * For a device guard which permits uninitialized state, see
 * InlineMaybeDeviceGuard.
 */
template <typename T>
class InlineDeviceGuard {
public:
  // Note [Omitted default constructor from RAII]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // In principle, we could add a default constructor to
  // DeviceGuard which reads the current device and promises to
  // restore to that device on exit.  However, most cases where you
  // would have written this, you probably meant to actually just
  // use MaybeDeviceGuard (since you don't actually need the
  // restore to happen if you don't ever actually set the device).
  // We remove the constructor here to encourage you to think about
  // what you actually want to happen.

  /// Set the current device to the passed Device.
  explicit InlineDeviceGuard(Device device)
    : impl_(device.type())
    , original_device_(device.index() == -1 ? impl_.getDevice() : impl_.exchangeDevice(device))
    , current_device_(device.index() == -1 ? original_device_ : device)
    {}

  /// Set the current device index to the passed DeviceIndex.  (The
  /// device type is inferred from the template parameter T).
  ///
  /// This constructor is only available for specializations that use a
  /// DeviceGuardImpl whose device type is statically known, e.g.,
  /// which define the static_type member.
  template <typename U=T, typename=std::enable_if<std::is_same<decltype(U::static_type), DeviceType>::value >>
  explicit InlineDeviceGuard(DeviceIndex device_index)
    : InlineDeviceGuard(Device(U::static_type, device_index)) {}

  // This constructor exists purely for testing
  template <typename U=T, typename=std::enable_if<std::is_same<U, VirtualGuardImpl>::value >>
  explicit InlineDeviceGuard(Device device, const DeviceGuardImplInterface* impl)
    : impl_(VirtualGuardImpl(impl))
    , original_device_(device.index() == -1 ? impl_.getDevice() : impl_.exchangeDevice(device))
    , current_device_(device.index() == -1 ? original_device_ : device)
    {}

  /// Copy is disallowed
  InlineDeviceGuard(const InlineDeviceGuard<T>&) = delete;
  InlineDeviceGuard<T>& operator=(const InlineDeviceGuard<T>&) = delete;

  /// Move is disallowed, as DeviceGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  InlineDeviceGuard(InlineDeviceGuard<T>&& other) = delete;
  InlineDeviceGuard& operator=(InlineDeviceGuard<T>&& other) = delete;

  ~InlineDeviceGuard() {
    impl_.uncheckedSetDevice(original_device_);
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    auto index = device.index();
    if (index == -1) return;
    AT_ASSERT(device.type() == original_device_.type());
    impl_.setDevice(device);
    current_device_ = device;
  }

  /// Sets the device index to the given one.
  void set_index(DeviceIndex index) {
    set_device(Device(original_device_.type(), index));
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() const {
    return original_device_;
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  Device current_device() const {
    return current_device_;
  }

protected:
  T impl_;

private:
  Device original_device_;
  Device current_device_;
};

/**
 * A MaybeDeviceGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * Morally, a MaybeDeviceGuard is equivalent to optional<DeviceGuard>, but
 * some methods are implemented more efficiently.
 *
 * Unlike DeviceGuard, a MaybeDeviceGuard may be uninitialized.  This occurs
 * when you use the nullary constructor, or pass a nullopt to the constructor.
 * Uninitialized MaybeDeviceGuards do *nothing*; they do not know what the
 * original device was, and they do not reset on destruction.
 *
 * An initialized InlineDeviceGuard doesn't restore device to its value at
 * construction; it restores device to its value *at initialization*.  So if you
 * have the program:
 *
 *     setDevice(1);
 *     Guard g;
 *     setDevice(2);
 *     g.set_device(3);
 *
 * On destruction, g will reset device to 2, rather than 1.
 *
 * An uninitialized MaybeDeviceGuard is distinct from a (initialized)
 * DeviceGuard whose original_device_ and current_device_ match, since the
 * DeviceGuard will still reset the device to original_device_.
 */
template <typename T>
class InlineMaybeDeviceGuard {
public:
  // Note [Explicit initialization of optional fields]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Explicit initialization of optional fields
  // required to workaround an nvcc bug; see https://github.com/pytorch/pytorch/issues/12117


  /// Default constructor, reads the current device so that
  /// we may reset the device to the current device on destruction.
  explicit InlineMaybeDeviceGuard()
    : guard_() // See Note [Explicit initialization of optional fields]
    {}

  /// Set the current device to the passed Device
  explicit InlineMaybeDeviceGuard(optional<Device> device_opt)
    : guard_() {
    if (device_opt.has_value()) {
      guard_.emplace(device_opt.value());
    }
  }

  /// All constructors of DeviceGuard are valid for MaybeDeviceGuard
  template <typename... Args>
  explicit InlineMaybeDeviceGuard(Args&&... args)
    : guard_(in_place, std::forward<Args>(args)...) {}

  // Note [Move construction for RAII guards is tricky]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // In principle, move construction is useful for terminating
  // the lifetime of a `MaybeDeviceGuard` early; for example:
  //
  //     // current device is d0
  //     MaybeDeviceGuard g1(d1);
  //     // current device is d1
  //     {
  //       MaybeDeviceGuard g2(std::move(g1));
  //     }
  //     // current device is d0!!
  //
  // However, it's difficult to implement the move constructor
  // in a way that works in all situations.  For example, consider
  // the following example:
  //
  //     MaybeDeviceGuard g1(d1);
  //     {
  //       MaybeDeviceGuard g2(d2);
  //       {
  //         MaybeDeviceGuard g3(std::move(g1)); // !!!
  //       }
  //     }
  //
  // What should the current device be while g3 in scope... and what
  // should it be after it goes out of scope?  What about g2?
  // There don't seem to be satisfactory answers for these questions.
  //
  // It's in principle possible to raise an error when this occurs
  // by doing some extra thread-local bookkeeping.  But why bother?
  // Just don't provide the constructor.
  InlineMaybeDeviceGuard(InlineMaybeDeviceGuard<T>&& other) = delete;

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
  // However, in example 1, this is determined by the restore value of g1
  // (prior to the move). In example 2, however, it is determined by the the
  // restore value of g2(!!). We don't know which one should win, without having
  // a way of telling which guard was allocated first.
  //
  // We could solve this with an extra thread-local variable.  But no one is
  // actually using move-assignment.  So just get rid of it.
  InlineMaybeDeviceGuard& operator=(InlineMaybeDeviceGuard&& other) = delete;

  /// Sets the device to the given one.  Initializes MaybeDeviceGuard if it
  /// is not already initialized.
  void set_device(at::Device device) {
    if (!guard_.has_value()) {
      guard_.emplace(device);
    } else {
      guard_->set_device(device);
    }
  }

  /// Sets the device index to the given one.
  ///
  /// This method is only available for specializations that use a
  /// DeviceGuardImpl whose device type is statically known, e.g.,
  /// which define the static_type member.
  template <typename U=T, typename=std::enable_if<std::is_same<decltype(U::static_type), DeviceType>::value >>
  void set_index(DeviceIndex index) {
    if (!guard_.has_value()) {
      guard_.emplace(index);
    } else {
      guard_->set_index(index);
    }
  }

  /// Returns the device that was set at the time the guard was initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.has_value() ? make_optional(guard_->original_device()) : nullopt;
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.has_value() ? make_optional(guard_->current_device()) : nullopt;
  }

private:
  optional<InlineDeviceGuard<T>> guard_;
};

}} // namespace c10::detail
