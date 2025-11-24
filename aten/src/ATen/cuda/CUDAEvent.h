#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

namespace at::cuda {

// EventPool - Thread-safe pool of CUDA events to avoid expensive
// cudaEventCreate calls. cudaEventCreate when concurrently invoked from
// multiple threads can be very expensive (especially on certain device/driver
// combinations).
using CUDAEventPtr =
    std::unique_ptr<CUDAEvent, std::function<void(CUDAEvent*)>>;

class EventPool {
 public:
  EventPool() : pools_(at::cuda::device_count()) {}

  CUDAEventPtr get(const DeviceIndex device) {
    // If the device is invalid, return a default event and no pooling
    if (device < 0 || device >= (DeviceIndex)pools_.size()) {
      auto deleter = [](CUDAEvent* event) {
        delete event;
      };
      return CUDAEventPtr(
        std::make_unique<CUDAEvent>(cudaEventDisableTiming).release(), deleter);
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
        return CUDAEventPtr(event.release(), destructor);
      }
    }

    return CUDAEventPtr(
        std::make_unique<CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

  void init_num_events(const size_t num_events) {
    for (DeviceIndex device_idx = 0; device_idx < at::cuda::device_count(); ++device_idx) {
        CUDAGuard device_guard(device_idx);
        std::vector<CUDAEventPtr> temp_events;
        temp_events.reserve(num_events);
        for (size_t i = 0; i < num_events; ++i) {
          auto event = get(device_idx);
          // Record the event to ensure it's properly initialized
          event->record();
          temp_events.emplace_back(std::move(event));
        }
        // Events will be returned to pool when temp_events is destroyed
    }
  }

 private:
  struct alignas(64) PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<CUDAEvent>> event_pool_;
  };

  std::vector<PerDevicePool> pools_;
};

} // namespace at::cuda
