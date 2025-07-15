#pragma once

#include <c10/hip/HIPEvent.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// See Note [Masquerading as CUDA] for motivation

struct HIPEventMasqueradingAsCUDA {
  HIPEventMasqueradingAsCUDA() noexcept = default;
  HIPEventMasqueradingAsCUDA(unsigned int flags) noexcept
      : event_(HIPEvent(flags)) {}
  HIPEventMasqueradingAsCUDA(
      DeviceIndex device_index,
      const hipIpcEventHandle_t* handle)
      : event_(HIPEvent(device_index, handle)) {}

  ~HIPEventMasqueradingAsCUDA() = default;

  HIPEventMasqueradingAsCUDA(const HIPEventMasqueradingAsCUDA&) = delete;
  HIPEventMasqueradingAsCUDA& operator=(const HIPEventMasqueradingAsCUDA&) = delete;
  HIPEventMasqueradingAsCUDA(HIPEventMasqueradingAsCUDA&& other) noexcept = default;
  HIPEventMasqueradingAsCUDA& operator=(HIPEventMasqueradingAsCUDA&& other) noexcept = default;

  operator hipEvent_t() const {
    return event_.event();
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(
      const HIPEventMasqueradingAsCUDA& left,
      const HIPEventMasqueradingAsCUDA& right) {
    return left.event_ < right.event_;
  }

  std::optional<c10::Device> device() const {
    // Unsafely coerce HIP device into CUDA device
    return Device(c10::DeviceType::CUDA, event_.device_index());
  }
  bool isCreated() const {
    return event_.isCreated();
  }
  DeviceIndex device_index() const {
    return event_.device_index();
  }
  hipEvent_t event() const {
    return event_.event();
  }
  bool query() const {
    return event_.query();
  }
  void record() {
    return event_.record();
  }

  void recordOnce(const HIPStreamMasqueradingAsCUDA& stream) {
    event_.recordOnce(stream.hip_stream());
  }

  void record(const HIPStreamMasqueradingAsCUDA& stream) {
    event_.record(stream.hip_stream());
  }

  void block(const HIPStreamMasqueradingAsCUDA& stream) {
    event_.block(stream.hip_stream());
  }

  float elapsed_time(const HIPEventMasqueradingAsCUDA& other) const {
    return event_.elapsed_time(other.event_);
  }

  void synchronize() const {
    event_.synchronize();
  }

  void ipc_handle(hipIpcEventHandle_t* handle) {
    event_.ipc_handle(handle);
  }

 private:
  HIPEvent event_;
};

}} // namespace c10::hip
