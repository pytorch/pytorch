#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

namespace at::cuda {

// CUDAEventPool - A thread-safe pool of CUDA events to avoid the overhead of
// repeatedly calling cudaEventCreate(). Concurrent cudaEventCreate() calls
// can incur significant cost on some device/driver combinations.
//
// This pool maintains per-device lists of pre-created CUDA events.
// Borrowed events are returned to the pool via a custom unique_ptr deleter.

class CUDAEventPool {
 public:
 using Event = std::unique_ptr<
      c10::cuda::CUDAEvent,
      std::function<void(c10::cuda::CUDAEvent*)>>;

  CUDAEventPool(size_t init_num_events = 0) : pools_(c10::cuda::device_count()) {
    if (init_num_events > 0) {
      reserve_events_on_pools(init_num_events);
    }
  }

  // Acquire an event associated with a given device. If device is invalid, fall
  // back to a regular CUDAEvent and no pooling.
  Event get(const DeviceIndex device) {
    if (device < 0 || device >= (DeviceIndex)pools_.size()) {
      auto deleter = [](CUDAEvent* event) {
        delete event;
      };
    return Event(
        std::make_unique<CUDAEvent>().release(), deleter);
    }

    auto& pool = pools_[device];

    // Create a destructor that returns the event to the appropriate device pool
    auto destructor = [&pool](CUDAEvent* event) noexcept {
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
        std::make_unique<CUDAEvent>().release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  // Pre-initialize each device pool with N events. This prevents
  // cudaEventCreate() from invoking during steady-state execution.
  void reserve_events_on_pools(size_t num_events) {
    for (const auto device : c10::irange(pools_.size())) {
      std::vector<Event> temp_events;
      temp_events.reserve(num_events);
      pools_[device].event_pool_.reserve(num_events);
      for([[maybe_unused]] const auto _ : c10::irange(num_events)) {
        auto event = get(device);
        event->create(device);
        temp_events.emplace_back(std::move(event));
      }
      // Events will be returned to pool when temp_events is destroyed.
    }
  }

  struct alignas(c10::hardware_destructive_interference_size) PerDevicePool {
    alignas(c10::hardware_destructive_interference_size) std::mutex mutex_;
    std::vector<std::unique_ptr<CUDAEvent>> event_pool_;
  };

  std::vector<PerDevicePool> pools_;
};

} // namespace at::cuda
