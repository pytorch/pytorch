#pragma once

#include <atomic>
#include <functional>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

namespace c10::cuda {

// forward declare CUDAAllocator
namespace CUDACachingAllocator {
class CUDAAllocator;
} // namespace CUDACachingAllocator

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the CUDACachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct C10_CUDA_API MemPool {
  MemPool(
      CUDACachingAllocator::CUDAAllocator* allocator = nullptr,
      bool is_user_created = true);

  MempoolId_t id();
  CUDACachingAllocator::CUDAAllocator* allocator();

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  CUDACachingAllocator::CUDAAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
};

// MemPoolContext holds the currently active pool and stashes the previous
// pool. On deletion it makes the previous pool active.
struct C10_CUDA_API MemPoolContext {
  MemPoolContext(MemPool* mempool);

  ~MemPoolContext();

  // getActiveMemPool() can be used to get the currently active pool.
  // For instance: in CUDACachingAllocator, we can route allocations
  // to a user provided allocator, by doing:
  //
  //  auto active_pool = MemPoolContext::getActiveMemPool();
  //  if (active_pool && active_pool->allocator()) {
  //    ptr = active_pool->allocator()->raw_alloc(size);
  //  }
  //
  static MemPool* getActiveMemPool();

 private:
  MemPool* prev_mempool_;
  static thread_local MemPool* active_mempool_;
};

} // namespace c10::cuda
