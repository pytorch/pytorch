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

  void create(DeviceIndex device_index) {
    event_.create(device_index);
  }

 private:
  HIPEvent event_;
};

class HIPEventPoolMasqueradingAsCUDA {
 public:
  using Event = std::unique_ptr<
      HIPEventMasqueradingAsCUDA,
      std::function<void(HIPEventMasqueradingAsCUDA*)>>;

  HIPEventPoolMasqueradingAsCUDA(size_t init_num_events = 0)
      : pools_(c10::hip::device_count()) {
    if (init_num_events > 0) {
      reserve_events_on_pools(init_num_events);
    }
  }

  Event get(DeviceIndex device) {
    if (device < 0 || device >= (DeviceIndex)pools_.size()) {
      auto deleter = [](HIPEventMasqueradingAsCUDA* event) { delete event; };
      return Event(
          std::make_unique<HIPEventMasqueradingAsCUDA>().release(), deleter);
    }

    auto& pool = pools_[device];

    // Create a destructor that returns the event to the appropriate device pool
    auto destructor = [&pool](HIPEventMasqueradingAsCUDA* event) noexcept {
      if (event != nullptr) {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        pool.event_pool_.emplace_back(event);
      }
    };

    {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto event = std::move(pool.event_pool_.back());
        pool.event_pool_.pop_back();
        return Event(event.release(), destructor);
      }
    }

    // Pool is empty then create a new Event
    return Event(
        std::make_unique<HIPEventMasqueradingAsCUDA>().release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  void reserve_events_on_pools(size_t num_events) {
    for (const auto device : c10::irange(pools_.size())) {
      std::vector<Event> temp_events;
      temp_events.reserve(num_events);
      pools_[device].event_pool_.reserve(num_events);
      for ([[maybe_unused]] const auto _ : c10::irange(num_events)) {
        auto event = get(device);
        event->create(device);
        temp_events.emplace_back(std::move(event));
      }
      // Events will be returned to pool when temp_events is destroyed.
    }
  }

  struct alignas(c10::hardware_destructive_interference_size) PerDevicePool {
    alignas(c10::hardware_destructive_interference_size) std::mutex mutex_;
    std::vector<std::unique_ptr<HIPEventMasqueradingAsCUDA>> event_pool_;
  };

  std::vector<PerDevicePool> pools_;
};

}} // namespace c10::hip
