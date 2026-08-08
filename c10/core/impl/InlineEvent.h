#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/InlineEventBase.h>
#include <c10/util/Exception.h>

namespace c10::impl {

template <typename T>
struct InlineEvent final : public InlineEventBase {
  InlineEvent() = delete;
  InlineEvent(
      const DeviceType _device_type,
      const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : InlineEventBase(_device_type, _flag), backend_{_device_type} {}

  // Copy constructor and copy assignment operator (deleted)
  InlineEvent(const InlineEvent&) = delete;
  InlineEvent& operator=(const InlineEvent&) = delete;

  // Move constructor and move assignment operator
  InlineEvent(InlineEvent&& other) noexcept
      : InlineEventBase(std::move(other)),
        backend_(std::move(other.backend_)),
        was_marked_for_recording_(other.was_marked_for_recording_) {
    other.was_marked_for_recording_ = false;
  }
  InlineEvent& operator=(InlineEvent&& other) noexcept {
    swap(other);
    return *this;
  }

  void swap(InlineEvent& other) noexcept {
    using std::swap;
    swap(
        static_cast<InlineEventBase&>(*this),
        static_cast<InlineEventBase&>(other));
    swap(backend_, other.backend_);
    swap(was_marked_for_recording_, other.was_marked_for_recording_);
  }

  ~InlineEvent() noexcept {
    if (event())
      base().destroyEvent(*this);
  }

  bool was_marked_for_recording() const noexcept {
    return was_marked_for_recording_;
  }

  void recordOnce(const Stream& stream) {
    if (!was_marked_for_recording_)
      record(stream);
  }

  void record(const Stream& stream) {
    TORCH_CHECK(
        stream.device_type() == device_type(),
        "Event device type ",
        DeviceTypeName(device_type()),
        " does not match recording stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    base().record(*this, stream);
    was_marked_for_recording_ = true;
    setDeviceIndex(stream.device_index());
  }

  void block(const Stream& stream) const {
    if (!was_marked_for_recording_)
      return;

    TORCH_CHECK(
        stream.device_type() == device_type(),
        "Event device type ",
        DeviceTypeName(device_type()),
        " does not match blocking stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    base().block(*this, stream);
  }

  bool query() const {
    if (!was_marked_for_recording_)
      return true;
    return base().queryEvent(*this);
  }

  void* eventId() const {
    return event();
  }

  double elapsedTime(const InlineEvent& other) const {
    TORCH_CHECK(
        other.device_type() == device_type(),
        "Event device type ",
        DeviceTypeName(device_type()),
        " does not match other's device type ",
        DeviceTypeName(other.device_type()),
        ".");
    TORCH_CHECK_VALUE(
        (flag() == EventFlag::BACKEND_DEFAULT) &&
            (other.flag() == EventFlag::BACKEND_DEFAULT),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        was_marked_for_recording() && other.was_marked_for_recording(),
        "Both events must be recorded before calculating elapsed time.");
    // elapsedTime in MPS can wait event to be completed if event is not ready,
    // which is a little different from CUDA
    TORCH_CHECK(
        (query() && other.query()) || device_type() == DeviceType::MPS,
        "Both events must be completed before calculating elapsed time.");

    return base().elapsedTime(*this, other);
  }

  void synchronize() const {
    if (!was_marked_for_recording_)
      return;
    base().synchronizeEvent(*this);
  }

 private:
  DeviceGuardImplInterface& base() {
    return static_cast<DeviceGuardImplInterface&>(backend_);
  }
  const DeviceGuardImplInterface& base() const {
    return static_cast<const DeviceGuardImplInterface&>(backend_);
  }

  T backend_;
  bool was_marked_for_recording_ = false;
};

} // namespace c10::impl
