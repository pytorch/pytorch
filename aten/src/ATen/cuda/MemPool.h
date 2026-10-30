#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <memory>

namespace at::cuda {

// Keep BC only
using c10::CaptureId_t;
using c10::MempoolId_t;

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the CUDACachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct TORCH_CUDA_CPP_API MemPool {
  MemPool(
      std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator =
          nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
  bool routes_pinned_ = false;

 public:
  // Lets threads route pinned allocations to a host pool with this pool's id
  // (use_mem_pool(pool, pinned=True)). The host pool lives as long as this.
  void enable_pinned();
};

// HostMemPool is a pool of pinned host memory in the CUDA host caching
// allocator. A thread routes its pinned allocations to the pool with
// at::HostAllocator::begin_allocate_current_thread_to_pool. Once the pool is
// destroyed and no thread is routed to it, its cached memory is freed, unless
// a graph capture is underway; what remains is freed by a later host
// empty_cache().
struct TORCH_CUDA_CPP_API HostMemPool {
  HostMemPool();
  HostMemPool(const HostMemPool&) = delete;
  HostMemPool(HostMemPool&&) = delete;
  HostMemPool& operator=(const HostMemPool&) = delete;
  HostMemPool& operator=(HostMemPool&&) = delete;
  ~HostMemPool();

  MempoolId_t id() const;
  int use_count() const;

 private:
  MempoolId_t id_;
};

} // namespace at::cuda
