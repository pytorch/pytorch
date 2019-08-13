#pragma once

#include "c10/core/impl/EventInterface.h"
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
 * impl::EventInterface should be used by other classes that work with
 * generic events.
 *
 * These generic events are built on top of DeviceGuardImpls, analogous
 * to DeviceGuard and InlineDeviceGuard. The name "DeviceGuardImpls,"
 * is no longer entirely accurate, as these classes implement the
 * backend-specific logic for a generic backend interface.
 *
 */

struct Event final : public impl::EventInterface {
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

  void recordOnce(const Stream& stream) {
    impl_.recordOnce(stream);
  }
  void record(const Stream& stream) {
    impl_.record(stream);
  }
  void block(const Stream& stream) const {
    impl_.block(stream);
  }
  bool query() const {
    return impl_.query();
  }

private:
  impl::InlineEvent<impl::VirtualGuardImpl> impl_;
};

} // c10
