#pragma once

#include <include/openreg.h>

#include "OpenRegException.h"
#include "OpenRegStream.h"

namespace c10::openreg {

struct OpenRegEvent {
  OpenRegEvent(bool enable_timing) noexcept : enable_timing_{enable_timing} {}

  ~OpenRegEvent() {
    if (is_created_) {
      OPENREG_CHECK(orEventDestroy(event_));
    }
  }

  OpenRegEvent(const OpenRegEvent&) = delete;
  OpenRegEvent& operator=(const OpenRegEvent&) = delete;

  OpenRegEvent(OpenRegEvent&& other) noexcept {
    moveHelper(std::move(other));
  }
  OpenRegEvent& operator=(OpenRegEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator orEvent_t() const {
    return event();
  }

  std::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kPrivateUse1, device_index_);
    } else {
      return std::nullopt;
    }
  }

  bool isCreated() const {
    return is_created_;
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

  orEvent_t event() const {
    return event_;
  }

  bool query() const {
    if (!is_created_) {
      return true;
    }

    orError_t err = orEventQuery(event_);
    if (err == orSuccess) {
      return true;
    }

    return false;
  }

  void record() {
    record(getCurrentOpenRegStream());
  }

  void recordOnce(const OpenRegStream& stream) {
    if (!was_recorded_)
      record(stream);
  }

  void record(const OpenRegStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");

    OPENREG_CHECK(orEventRecord(event_, stream));
    was_recorded_ = true;
  }

  void block(const OpenRegStream& stream) {
    if (is_created_) {
      OPENREG_CHECK(orStreamWaitEvent(stream, event_, 0));
    }
  }

  float elapsed_time(const OpenRegEvent& other) const {
    TORCH_CHECK_VALUE(
        !(enable_timing_ & orEventDisableTiming) &&
            !(other.enable_timing_ & orEventDisableTiming),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");

    float time_ms = 0;
    OPENREG_CHECK(orEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      OPENREG_CHECK(orEventSynchronize(event_));
    }
  }

 private:
  unsigned int enable_timing_{orEventDisableTiming};
  bool is_created_{false};
  bool was_recorded_{false};
  DeviceIndex device_index_{-1};
  orEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    OPENREG_CHECK(orEventCreateWithFlags(&event_, enable_timing_));
    is_created_ = true;
  }

  void moveHelper(OpenRegEvent&& other) {
    std::swap(enable_timing_, other.enable_timing_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10::openreg
