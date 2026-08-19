#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

namespace c10::impl {

/*
 * Note [Event Semantics]
 *
 * `event_` is lazily initialized in one of three ways:
 *   1. on the first record(),
 *   2. via reconstructFromIPCHandle() (importer), or
 *   3. via ipcHandle() (exporter).
 *
 * block(), query(), and synchronize() therefore guard on event_ != nullptr
 * rather than on `was_marked_for_recording_`, which only tracks whether
 * record() was called in the current process.
 *
 * The IPC importer calls reconstructFromIPCHandle() to initialize `event_`
 * and resets `flag_` to PYTORCH_DEFAULT to prevent re-export via ipcHandle().
 */

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
  InlineEvent(InlineEvent&& other) noexcept
      : event_(other.event_),
        backend_(std::move(other.backend_)),
        device_type_(other.device_type_),
        device_index_(other.device_index_),
        flag_(other.flag_),
        was_marked_for_recording_(other.was_marked_for_recording_) {
    other.event_ = nullptr;
  }
  InlineEvent& operator=(InlineEvent&& other) noexcept {
    swap(other);
    return *this;
  }

  void swap(InlineEvent& other) noexcept {
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
    // See Note [Event Semantics]
    if (!event_)
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
    // See Note [Event Semantics]
    if (!event_)
      return true;
    return backend_.queryEvent(event_);
  }

  void* eventId() const {
    return event_;
  }

  double elapsedTime(const InlineEvent& other) const {
    TORCH_CHECK(
        other.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match other's device type ",
        DeviceTypeName(other.device_type()),
        ".");
    TORCH_CHECK_VALUE(
        (flag_ & EventFlag::TIMING) && (other.flag_ & EventFlag::TIMING),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        was_marked_for_recording() && other.was_marked_for_recording(),
        "Both events must be recorded before calculating elapsed time.");
    // elapsedTime in MPS can wait event to be completed if event is not ready,
    // which is a little different from CUDA
    TORCH_CHECK(
        (query() && other.query()) || device_type_ == DeviceType::MPS,
        "Both events must be completed before calculating elapsed time.");

    return backend_.elapsedTime(event_, other.event_, device_index_);
  }

  void synchronize() const {
    // See Note [Event Semantics]
    if (!event_)
      return;
    backend_.synchronizeEvent(event_);
  }

  std::string ipcHandle() {
    TORCH_CHECK(flag_ & EventFlag::INTERPROCESS, "Event is not an IPC event.");
    // See Note [Event Semantics]: event_ may be lazily initialized on the
    // current device if record() was never called.
    if (device_index_ == -1) {
      device_index_ = backend_.getDevice().index();
    }
    return backend_.getEventIPCHandle(&event_, device_index_, flag_);
  }

  void reconstructFromIPCHandle(
      const DeviceIndex device_index,
      const std::string& handle_string) {
    TORCH_CHECK(
        flag_ & EventFlag::INTERPROCESS,
        "Event must be created with EventFlag::INTERPROCESS to reconstruct from an IPC handle.");
    TORCH_CHECK(
        !event_,
        "Event has already been initialized; cannot reconstruct from an IPC handle.");
    device_index_ =
        device_index == -1 ? backend_.getDevice().index() : device_index;
    backend_.reconstructEventFromIPCHandle(
        &event_, device_index_, handle_string);
    // See Note [Event Semantics]
    flag_ = EventFlag::PYTORCH_DEFAULT;
  }

 private:
  void* event_ = nullptr;
  T backend_;
  DeviceType device_type_;
  DeviceIndex device_index_ = -1;
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  bool was_marked_for_recording_ = false;
};

} // namespace c10::impl
