#pragma once

#include <c10/Device.h>
#include <c10/DeviceType.h>
#include <c10/Stream.h>

// Just for C10_ANONYMOUS_VARIABLE
#include <c10/util/Registry.h>

#include <atomic>

namespace c10 {
namespace detail {

/**
 * DeviceGuardImplInterface represents the virtual interface which provides
 * functionality to provide an RAII class for device and stream switching,
 * via DeviceGuard.  Every distinct device type, e.g., CUDA and HIP, is
 * expected to implement and register an implementation of this interface.
 * All classes which inherit from DeviceGuardImplInterface should be declared
 * 'final'.
 *
 * This class exists because we provide a unified interface for performing
 * device guards via DeviceGuard, but we cannot assume that we have actually
 * compiled against the, e.g., CUDA library, which actually implements
 * this guard functionality.  In this case, a dynamic dispatch is required
 * to cross the library boundary.
 *
 * If possible, you should directly use implementations of this interface;
 * those uses will be devirtualized.
 */
struct C10_API DeviceGuardImplInterface {
  /**
   * Return the type of device managed by this guard implementation.
   */
  virtual DeviceType type() const = 0;

  /**
   * Set the current device to Device, and return the previous Device.
   */
  virtual Device exchangeDevice(Device) const = 0;
  // NB: Implementations of exchangeDevice can be a bit boilerplatey.  You might
  // consider replacing exchangeDevice with a non-virtual function with a baked
  // in implementation; however, note that this will triple the number of
  // virtual calls (when you implement exchangeDevice in a final subclass,
  // the compiler gets to devirtualize everything; it won't do that if you don't
  // define it in the subclass!)  A common way to solve this problem is to use
  // some sort of CRTP; however, we can template DeviceGuardImplInterface since
  // we really *do* need it to be virtual.  A little boilerplate seems easiest
  // to explain.  (Another way around this problem is to provide inline
  // functions that provide the default implementations, but this seems a little
  // hard to explain.  In any case, we're only going to have on order of ten
  // implementations of this anyway.)

  /**
   * Get the current device.
   */
  virtual Device getDevice() const = 0;

  /**
   * Set the current device to Device.
   */
  virtual void setDevice(Device) const = 0;

  /**
   * Set the current device to Device, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  virtual void uncheckedSetDevice(Device) const noexcept = 0;

  /**
   * Get the current stream for a given device.
   */
  virtual Stream getStream(Device) const noexcept = 0;

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  virtual Stream exchangeStream(Stream) const noexcept = 0;

  /**
   * Intended use of this class is to leak the DeviceGuardImpl at program end.
   * So you better not call the destructor, buster!
   */
  virtual ~DeviceGuardImplInterface() = default;
};

// The registry is NON-owning.  Each stored pointer is std::atomic so
// that under all interleavings of registry calls the structure is
// race-free.  This doesn't cost us anything on reads in X86.  (An
// unsynchronized implementation probably is OK too, but I didn't want
// to prove that we never read from device_guard_impl_registry at the
// same time some registration is occurring.  Shiver.)
//
// I'd like this registry to be valid even at program destruction time
// (in case someone uses a DeviceGuard in a destructor to do some cleanup
// in the CUDA API.)  Since there are no direct accesses of the underlying
// owning objects which I can use to enforce initialization order (unlike
// in a Meyer singleton), it implies that you must *leak* objects when
// putting them in the registry.  This is done by deleting the destructor
// on DeviceGuardImplInterface.
extern C10_API std::atomic<const DeviceGuardImplInterface*>
device_guard_impl_registry[static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

// I can't conveniently use c10/util/Registry.h for the following reason:
// c10/util/Registry.h gives me a slow way of Create'ing a object of some
// interface from the registry, but no way of quickly accessing an already
// created object.  I'll be banging on getDeviceGuardImpl every time we do a
// DeviceGuard, so I really don't want to be doing an unordered_map lookup.
// Better if the registration mechanism directly drops its implementation
// into device_guard_impl_registry.

class C10_API DeviceGuardImplRegistrar {
public:
  DeviceGuardImplRegistrar(DeviceType, const DeviceGuardImplInterface*);
};

#define C10_REGISTER_GUARD_IMPL(DevType, DeviceGuardImpl) \
  static ::c10::detail::DeviceGuardImplRegistrar C10_ANONYMOUS_VARIABLE(g_##DeviceType)(::c10::DeviceType::DevType, new DeviceGuardImpl());

inline const DeviceGuardImplInterface* getDeviceGuardImpl(DeviceType type) {
  auto p = device_guard_impl_registry[static_cast<size_t>(type)].load();
  AT_ASSERTM(p, "DeviceGuardImpl for ", type, " is not available");
  return p;
}

}} // namespace c10::detail
