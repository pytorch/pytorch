#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/EventFlag.h>

namespace c10::impl {

struct InlineEventBase {
  InlineEventBase(const InlineEventBase&) = delete;
  InlineEventBase& operator=(const InlineEventBase&) = delete;

  DeviceType device_type() const noexcept {
    return device_type_;
  }
  DeviceIndex device_index() const noexcept {
    return device_index_;
  }
  EventFlag flag() const noexcept {
    return flag_;
  }
  void* event() const noexcept {
    return event_;
  }
  void setEvent(void* p) noexcept {
    event_ = p;
  }
  void setDeviceIndex(const DeviceIndex i) noexcept {
    device_index_ = i;
  }

 protected:
  InlineEventBase(const DeviceType device_type, const EventFlag flag)
      : device_type_{device_type}, flag_{flag} {}

  InlineEventBase(InlineEventBase&& other) noexcept
      : event_(other.event_),
        device_type_(other.device_type_),
        device_index_(other.device_index_),
        flag_(other.flag_) {
    other.event_ = nullptr;
  }
  InlineEventBase& operator=(InlineEventBase&& other) noexcept = default;

  friend void swap(InlineEventBase& a, InlineEventBase& b) noexcept {
    using std::swap;
    swap(a.event_, b.event_);
    swap(a.device_type_, b.device_type_);
    swap(a.device_index_, b.device_index_);
    swap(a.flag_, b.flag_);
  }

 private:
  void* event_ = nullptr;
  DeviceType device_type_{};
  DeviceIndex device_index_ = -1;
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
};

} // namespace c10::impl
