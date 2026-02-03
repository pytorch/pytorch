#include "OpenRegHostAllocator.h"

#include <c10/util/Exception.h>

#include <mutex>
#include <unordered_map>

#include <unistd.h>

namespace c10::openreg {

namespace {

class HostMemoryAllocator {
 public:
  HostMemoryAllocator() = default;

  HostMemoryAllocator(const HostMemoryAllocator&) = delete;
  HostMemoryAllocator& operator=(const HostMemoryAllocator&) = delete;

  void* malloc(size_t nbytes) {
    if (nbytes == 0) {
      return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    static const size_t page_size_bytes = [] {
      constexpr size_t kDefaultPageSizeBytes = 4096;
      const long page_size_long = sysconf(_SC_PAGESIZE);
      return page_size_long > 0 ? static_cast<size_t>(page_size_long)
                                : kDefaultPageSizeBytes;
    }();
    const size_t aligned_nbytes =
        ((nbytes + page_size_bytes - 1) / page_size_bytes) * page_size_bytes;

    void* data = nullptr;
    const auto ret = orMallocHost(&data, aligned_nbytes);

    TORCH_CHECK(
        ret == orSuccess && data != nullptr,
        "Failed to allocate ",
        aligned_nbytes,
        " bytes on host.");

    allocation_sizes_[data] = aligned_nbytes;

    stats_.allocations.increase(1);
    stats_.allocated_bytes.increase(aligned_nbytes);
    stats_.active_requests.increase(1);
    stats_.active_bytes.increase(aligned_nbytes);

    stats_.num_host_alloc += 1;

    return data;
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    auto it = allocation_sizes_.find(ptr);
    if (it != allocation_sizes_.end()) {
      const size_t nbytes = it->second;
      const auto ret = orFreeHost(ptr);
      if (ret != orSuccess && ret != orErrorUnknown) {
        TORCH_WARN(
            "orFreeHost failed for OpenReg host pointer ",
            ptr,
            ". Return code: ",
            ret);
        return;
      }

      allocation_sizes_.erase(it);

      stats_.allocations.decrease(1);
      stats_.allocated_bytes.decrease(nbytes);
      stats_.active_requests.decrease(1);
      stats_.active_bytes.decrease(nbytes);

      if (ret == orSuccess) {
        stats_.num_host_free += 1;
      }

      return;
    }

    // Best-effort orFreeHost for untracked pointers; don't update stats.
    const auto ret = orFreeHost(ptr);
    if (ret != orSuccess && ret != orErrorUnknown) {
      TORCH_WARN(
          "orFreeHost failed for untracked OpenReg host pointer ",
          ptr,
          ". Return code: ",
          ret);
    }
  }

  at::HostStats getStats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return stats_;
  }

  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocations.reset_accumulated();
    stats_.allocated_bytes.reset_accumulated();
    stats_.active_requests.reset_accumulated();
    stats_.active_bytes.reset_accumulated();

    stats_.host_alloc_time.reset_accumulated();
    stats_.host_free_time.reset_accumulated();

    stats_.num_host_alloc = 0;
    stats_.num_host_free = 0;
  }

  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocations.reset_peak();
    stats_.allocated_bytes.reset_peak();
    stats_.active_requests.reset_peak();
    stats_.active_bytes.reset_peak();

    stats_.host_alloc_time.reset_peak();
    stats_.host_free_time.reset_peak();
  }

 private:
  mutable std::recursive_mutex mutex_;

  at::HostStats stats_{};
  std::unordered_map<void*, size_t> allocation_sizes_;
};

HostMemoryAllocator g_host_allocator;

void deleteOpenRegHostMemory(void* ptr) {
  g_host_allocator.free(ptr);
}

} // namespace

at::DataPtr OpenRegHostAllocator::allocate(size_t nbytes) {
  void* data = g_host_allocator.malloc(nbytes);
  return {data, data, &deleteOpenRegHostMemory, at::Device(at::kCPU)};
}

at::DeleterFnPtr OpenRegHostAllocator::raw_deleter() const {
  return &deleteOpenRegHostMemory;
}

void OpenRegHostAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  orMemcpy(dest, src, count, orMemcpyHostToHost);
}

bool OpenRegHostAllocator::record_event(void* ptr, void* ctx, c10::Stream stream) {
  (void)ptr;
  (void)ctx;
  (void)stream;
  return true;
}

void OpenRegHostAllocator::empty_cache() {
  // No caching for OpenReg host allocations today.
}

at::HostStats OpenRegHostAllocator::get_stats() {
  return g_host_allocator.getStats();
}

void OpenRegHostAllocator::reset_accumulated_stats() {
  g_host_allocator.resetAccumulatedStats();
}

void OpenRegHostAllocator::reset_peak_stats() {
  g_host_allocator.resetPeakStats();
}

OpenRegHostAllocator caching_host_allocator;
REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &caching_host_allocator);

} // namespace c10::openreg
