#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {

class CPUCachingAllocator {
  /*
   * What it does:
   * Caches all the allocations carried out by this allocator.
   * Cache key is the size of the allocation.
   * If requested size is found in the cache returns the cached pointer.
   * What it does not do:
   * No speculative allocation for any future allocations.
   */
  private:
    std::mutex mutex_;
    // Invariants.
    // 1. If memory is ever allocated via this allocator then
    //    the pointer will exist in allocation_map_, unless the allocator
    //    returned the memory to OS via free_cached.
    //  1.1. Therefore even when the said memory is freed via this
    //       allocator, it will continue to stay in allocaiton_map_.
    //       Furthermore it will also exist in available_map_.
    //       Thus an allocated memory pointer can be in both
    //       allocation_map_ and available_map_ simultaneously.
    // 2. Memory pointer maybe removed from allocation_map_, when it
    //    is freed outside of the scope of this allocator, but was allocated
    //    by this allocator.
    // 3. Available map only contains that memory which was allocated
    //    by this allocator and subsequently freed by this allocator.
    // As a result of above invariants, allocated memory ptr cannot be in
    // available_map_ unless it is in allocation_map_ as well.
    ska::flat_hash_map<void*, size_t> allocation_map_;
    // TODO:
    // Replace std::vector with c10::SmallVector
    // At the moment using that fails to compile due to the way
    // flat_hash_map is written.
    ska::flat_hash_map<size_t, std::vector<void*>> available_map_;
    inline void* allocate_and_cache(const size_t bytes);
    inline void* use_cached(const size_t bytes);
    void free_cached();
  public:
    // Checks the cache to see if allocation of size bytes can be found.
    // If so return cached memory, else
    // allocates memory, records it for caching and returns.
    void* allocate(const size_t bytes);
    // Checks if the memory being freed is was marked for allocation by
    // an earlier call to allocate. If so cache the allocation.
    // Otherwise free.
    void free(void* ptr);
    void record_free(void* ptr);
    // Mainly for testing
    ~CPUCachingAllocator();
};

CPUCachingAllocator& GetCPUCachingAllocator();

class C10_API CachingAllocatorInfo {
  public:
    void set(const CachingAllocatorInfo& other);
    bool enabled();
    void enable();
    void disable();
  private:
    bool is_enabled_{false};
};

CachingAllocatorInfo& GetThreadLocalCachingAllocatorInfo();

class C10_API WithCPUCachingAllocatorGuard {
  public:
    WithCPUCachingAllocatorGuard(bool enabled=true);
    ~WithCPUCachingAllocatorGuard();
  private:
    CachingAllocatorInfo prev_info_;
};

} // namespace c10
