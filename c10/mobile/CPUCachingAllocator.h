#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/flat_hash_map.h>

/*
 * CPUCachingAllocator:
 * DISCLAIMER:
 *    This is subject to change (beta) and only supported on mobile builds.
 *    If code snippet such as in 'Usage pattern' is used outside of mobile
 *    build you will not observe the intended behavior.
 *    See below for more information.
 * Why?
 *    It has been observed that some mobile platforms, such as pixel 3, return
 *    memory aggressively to the system. This results in page faults in some
 * cases and ends up hurting performance. This caching allocator aims to address
 * that. Furthermore it also allows users to specify their own allocator by
 * implementing allocate/free virtual interfaces. What are the cons? There are
 * some cons that were observed where use of caching allocator led to worse
 * performance on some platforms. Reason being that the caching mechanism used
 * by this allocator left us worse off compared to the corresponding platform's
 *    tuned memory allocator. In that case it seemed better to not use this
 * allocator. Note there are some ideas to fix this in the works.
 *
 * Usage:
 * Usage pattern:
 * Instantiate and own the caching allocator.
 * std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
 *   std::make_unique<c10::CPUCachingAllocator>();
 * Use caching allocator with a scoped guard at inference time.
 * {
 * WithCPUCachingAllocatorGuard(caching_allocator.get());
 * ... model.forward(...);
 * }
 */

namespace c10 {

class C10_API CPUCachingAllocator {
  /*
   * What it does:
   * Caches all the allocations carried out by this allocator.
   * Cache key is the size of the allocation.
   * If requested size is found in the cache returns the cached pointer.
   * What it does not do:
   * No speculative allocation for any future allocations.
   */
 private:
  inline void* allocate_and_cache(const size_t bytes);
  void free_cached();

 protected:
  // Invariants.
  // 1. If memory is ever allocated via this allocator then
  //    the pointer will exist in allocation_map_, unless the allocator
  //    returned the memory to OS via free_cached.
  //  1.1. Therefore even when the said memory is "freed" via this
  //       allocator (and thus cached), it will continue to stay
  //       in allocation_map_. Furthermore it will also exist in
  //       available_map_. Thus an allocated memory pointer can be in both
  //       allocation_map_ and available_map_ simultaneously.
  // 2. Memory pointer maybe removed from allocation_map_, when it
  //    is freed outside of the scope of this allocator, but was allocated
  //    by this allocator.
  // 3. Available map only contains that memory which was allocated
  //    by this allocator and subsequently freed by this allocator.
  // As a result of above invariants, allocated memory ptr cannot be in
  // available_map_ unless it is in allocation_map_ as well.
  ska::flat_hash_map<size_t, c10::SmallVector<void*, 16>> available_map_;
  static ska::flat_hash_map<void*, size_t> allocation_map_;
  // Since allocation_map, which is a global instance, is mutated/read via
  // all public APIs we need a global mutex.
  static std::mutex mutex_;

 public:
  static void record_free(void* ptr);
  virtual ~CPUCachingAllocator();
  // Checks the cache to see if allocation of size bytes can be found.
  // If so return cached memory, else
  // allocates memory, records it for caching and returns.
  virtual void* allocate(const size_t bytes);
  // Checks if the memory being freed is was marked for allocation by
  // an earlier call to allocate. If so cache the allocation.
  // Otherwise free.
  virtual void free(void* ptr);
};

CPUCachingAllocator* GetDefaultCPUCachingAllocator();

bool ThreadLocalCachingAllocatorEnabled();
CPUCachingAllocator* GetThreadLocalCachingAllocator();

class C10_API WithCPUCachingAllocatorGuard {
 public:
  WithCPUCachingAllocatorGuard(CPUCachingAllocator* allocator);
  ~WithCPUCachingAllocatorGuard();

 private:
  CPUCachingAllocator* prev_caching_allocator_ptr_{nullptr};
};

} // namespace c10
