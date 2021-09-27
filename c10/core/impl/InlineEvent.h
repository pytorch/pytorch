#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

namespace c10 {
namespace impl {

template <typename T>
struct InlineEvent final {
  InlineEvent() = delete;
  InlineEvent(
      const DeviceType _device_type,
      const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : backend_{_device_type}, device_type_{_device_type}, flag_{_flag} {}

  // Copy constructor and copy assignment operator (deleted)
  InlineEvent(const InlineEvent&) = delete;
  InlineEvent& operator=(const InlineEvent&) = delete;

  // Move constructor and move assignment operator
  InlineEvent(InlineEvent&& other)
      : InlineEvent(other.device_type_, other.flag_) {
    swap(std::move(other));
  }
  InlineEvent& operator=(InlineEvent&& other) {
    swap(std::move(other));
    return *this;
  }

  void swap(InlineEvent&& other) {
    std::swap(event_, other.event_);
    std::swap(backend_, other.backend_);
    std::swap(device_type_, other.device_type_);
    std::swap(device_index_, other.device_index_);
    std::swap(flag_, other.flag_);
    std::swap(was_marked_for_recording_, other.was_marked_for_recording_);
  }

  ~InlineEvent() noexcept {
    if (event_)
      backend_.destroyEvent(event_, device_index_);
  }

  DeviceType device_type() const noexcept {
    return device_type_;
  }
  DeviceIndex device_index() const noexcept {
    return device_index_;
  }
  EventFlag flag() const noexcept {
    return flag_;
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
        stream.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match recording stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    backend_.record(&event_, stream, device_index_, flag_);
    was_marked_for_recording_ = true;
    device_index_ = stream.device_index();
  }

  void block(const Stream& stream) const {
    if (!was_marked_for_recording_)
      return;

    TORCH_CHECK(
        stream.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match blocking stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    backend_.block(event_, stream);
  }

  bool query() const {
    if (!was_marked_for_recording_)
      return true;
    return backend_.queryEvent(event_);
  }

 private:
  void* event_ = nullptr;
  T backend_;
  DeviceType device_type_;
  DeviceIndex device_index_ = -1;
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  bool was_marked_for_recording_ = false;
};

} // namespace impl
} // namespace c10
