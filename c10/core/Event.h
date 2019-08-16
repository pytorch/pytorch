#pragma once

#include "c10/core/impl/InlineEvent.h"
#include "c10/core/impl/VirtualGuardImpl.h"

namespace c10 {

/**
 * A backend-generic movable, not copyable, not thread-safe event.
 *
 * Backend-generic events are implemented by this class and
 * impl::InlineEvent, both of which inherit from the impl::EventInterface
 * pure virtual class. In addition to these events there are also
 * some backend-specific events, like ATen's CUDAEvent. Each of these
 * classes has its own use.
 *
 * impl::InlineEvent<...> or a backend-specific event should be
 * preferred when the backend is known at compile time and known to
 * be compiled. Backend-specific events may have additional functionality.
 *
 * This Event should be used if a particular backend may not be available,
 * or the backend required is not known at compile time.
 *
 * These generic events are built on top of DeviceGuardImpls, analogous
 * to DeviceGuard and InlineDeviceGuard. The name "DeviceGuardImpls,"
 * is no longer entirely accurate, as these classes implement the
 * backend-specific logic for a generic backend interface.
 *
 * See DeviceGuardImplInterface.h for a list of all supported flags.
 */

struct Event final {
  // Constructors
  Event() = delete;
  Event(
    const DeviceType _device_type,
    const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
  : impl_{_device_type, _flag} { }

  // Copy constructor and copy assignment operator (deleted)
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  // Move constructor and move assignment operator
  Event(Event&& other) : impl_{std::move(other.impl_)} { }
  Event& operator=(Event&& other) {
  impl_.swap(std::move(other.impl_));
    return *this;
  }

  // Destructor
  ~Event() = default;

  // Getters
  DeviceType device_type() const noexcept { return impl_.device_type(); }
  DeviceIndex device_index() const noexcept { return impl_.device_index(); }
  EventFlag flag() const noexcept { return impl_.flag(); }
  bool was_marked_for_recording() const noexcept { return impl_.was_marked_for_recording(); }

/**
 * Calls record() if and only if record() has never been called for this event.
 * Note: because Event is not thread-safe recordOnce() may call record()
 * multiple times if called from multiple threads.
 */
  void recordOnce(const Stream& stream) {
    impl_.recordOnce(stream);
  }

/**
 * Marks the event as not recorded and enqueues the event in the
 * stream's work queue. When the stream processes the event either:
 *  (1) the event is marked as recorded
 *  (2) if the event was enqueued again, nothing happens
 * Put another way, events reflect only the most recent call to record.
 */
  void record(const Stream& stream) {
    impl_.record(stream);
  }

/**
 * Does nothing if the event has not been scheduled to be recorded.
 * If the event was previously enqueued to be recorded, a command
 * to wait for the event is inserted in the stream's work queue.
 * When the stream reaches this command it will stop processing
 * additional commands until the event is marked as recorded.
 */
  void block(const Stream& stream) const {
    impl_.block(stream);
  }

/**
 * Returns true if (and only if)
 *  (1) the event has never been scheduled to be recorded
 *  (2) is marked as recorded.
 * Returns false otherwise.
 */
  bool query() const {
    return impl_.query();
  }

private:
  impl::InlineEvent<impl::VirtualGuardImpl> impl_;
};

} // c10
