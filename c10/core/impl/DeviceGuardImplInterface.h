#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

// Just for C10_ANONYMOUS_VARIABLE
#include <c10/util/Registry.h>

#include <atomic>

namespace c10 {

// Forward declaration
class DataPtr;

/**
 * Flags defining the behavior of events.
 *
 * PYTORCH_DEFAULT and BACKEND_DEFAULT are valid for all backends. The
 * BACKEND_DEFAULT is what a particular backend would select if no
 * flags were given. PYTORCH_DEFAULT is the PyTorch's framework default
 * choice for events on that backend, which may not be the same.
 *
 * The mapping of PYTORCH_DEFAULT and BACKEND_DEFAULT is done by each
 * backend implementation.
 */
enum class EventFlag {
  // Disable timing
  PYTORCH_DEFAULT,
  // Enable timing
  BACKEND_DEFAULT,
  // FOR TESTING ONLY
  INVALID
};

namespace impl {

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
  DeviceGuardImplInterface() = default;
  DeviceGuardImplInterface(const DeviceGuardImplInterface&) = default;
  DeviceGuardImplInterface& operator=(const DeviceGuardImplInterface&) =
      default;
  DeviceGuardImplInterface(DeviceGuardImplInterface&&) noexcept = default;
  DeviceGuardImplInterface& operator=(DeviceGuardImplInterface&&) noexcept =
      default;

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
   * Get the default stream for a given device.
   */
  virtual Stream getDefaultStream(Device) const {
    TORCH_CHECK(false, "Backend doesn't support acquiring a default stream.")
  }

  /**
   * Get a stream from the global pool for a given device.
   */
  virtual Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const {
    (void)isHighPriority; // Suppress unused variable warning
    TORCH_CHECK(false, "Backend doesn't support acquiring a stream from pool.")
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  virtual Stream getNewStream(Device, int priority = 0) const {
    (void)priority;
    TORCH_CHECK(false, "Backend doesn't support create a new Stream.")
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  virtual Stream exchangeStream(Stream) const noexcept = 0;

  /**
   * Destroys the given event.
   */
  virtual void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept {}

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  virtual void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const c10::EventFlag /*flag*/) const {
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  virtual void block(void* /*event*/, const Stream& /*stream*/) const {
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  virtual bool queryEvent(void* /*event*/) const {
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  /**
   * Get the number of devices.  WARNING: This is REQUIRED to not raise
   * an exception.  If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  virtual DeviceIndex deviceCount() const noexcept = 0;

  /**
   * Return true if all the work previously enqueued on the stream for
   * asynchronous execution has completed running on the device.
   */
  virtual bool queryStream(const Stream& /*stream*/) const {
    TORCH_CHECK(false, "Backend doesn't support querying streams.");
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * enqueued on the stream has completed running on the device.
   */
  virtual void synchronizeStream(const Stream& /*stream*/) const {
    TORCH_CHECK(false, "Backend doesn't support synchronizing streams.");
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  virtual void synchronizeEvent(void* /*event*/) const {
    TORCH_CHECK(false, "Backend doesn't support synchronizing events.");
  }

  /**
   * Ensure the caching allocator (if any) is aware that the given DataPtr is
   * being used on the given stream, and that it should thus avoid recycling the
   * DataPtr until all work on that stream is done.
   */
  virtual void recordDataPtrOnStream(const c10::DataPtr&, const Stream&) const {
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  virtual double elapsedTime(void* /*event1*/, void* /*event2*/) const {
    TORCH_CHECK(false, "Backend doesn't support elapsedTime.");
  }

  /**
   * Intended use of this class is to leak the DeviceGuardImpl at program end.
   * So you better not call the destructor, buster!
   */
  virtual ~DeviceGuardImplInterface() = default;
};

// A no-op device guard impl that doesn't do anything interesting.  Useful
// for devices that don't actually have a concept of device index.  Prominent
// examples are CPU and Meta.
template <DeviceType D>
struct NoOpDeviceGuardImpl final : public DeviceGuardImplInterface {
  NoOpDeviceGuardImpl() = default;
  DeviceType type() const override {
    return D;
  }
  Device exchangeDevice(Device) const override {
    return Device(D, -1); // no-op
  }
  Device getDevice() const override {
    return Device(D, -1);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, -1));
  }

  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return Stream(Stream::DEFAULT, Device(D, -1));
  }

  // NB: These do NOT set the current device
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, -1));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}

  // Stream-related functions
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
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
// NOLINTNEXTLINE(*c-arrays*)
extern C10_API std::atomic<const DeviceGuardImplInterface*>
    device_guard_impl_registry[static_cast<size_t>(
        DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

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

#define C10_REGISTER_GUARD_IMPL(DevType, DeviceGuardImpl)              \
  static ::c10::impl::DeviceGuardImplRegistrar C10_ANONYMOUS_VARIABLE( \
      g_##DeviceType)(::c10::DeviceType::DevType, new DeviceGuardImpl());

inline const DeviceGuardImplInterface* getDeviceGuardImpl(DeviceType type) {
  // Two adjacent int16_t fields DeviceType and DeviceIndex has field access
  // miscompiled on NVCC. To workaround this issue, we apply a mask to the
  // DeviceType. First check if the DeviceType is 16-bit.
  // FB employees can see
  //   https://fb.workplace.com/groups/llvm.gcc/permalink/4053565044692080/
  // for more details
  static_assert(sizeof(DeviceType) == 1, "DeviceType is not 8-bit");
  auto p = device_guard_impl_registry[static_cast<size_t>(type) & 0xFF].load();

  // This seems to be the first place where you make use of a device
  // when you pass devices to factory functions.  Give a nicer error
  // message in this case.
  TORCH_CHECK(p, "PyTorch is not linked with support for ", type, " devices");
  return p;
}

inline bool hasDeviceGuardImpl(DeviceType type) {
  return device_guard_impl_registry[static_cast<size_t>(type)].load();
}

} // namespace impl
} // namespace c10
