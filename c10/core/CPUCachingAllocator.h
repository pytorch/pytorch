#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Exception.h>
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
    // 1. If an allocated memory pointer exists in allocation_map_ then
    //    it must not exist in available_map_, and vice-e-versa.
    // 2. All the pointers in available_map_ must be unique.
    ska::flat_hash_map<void*, size_t> allocation_map_;
    ska::flat_hash_map<size_t, std::deque<void*>> available_map_;
    inline void* allocate_and_cache(const size_t bytes);
    inline void* use_cached(const size_t bytes);
  public:
    void* allocate(const size_t bytes);
    void free(void* ptr);
    void free_cached();
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
    WithCPUCachingAllocatorGuard();
    ~WithCPUCachingAllocatorGuard();
  private:
    CachingAllocatorInfo prev_info_;
};

} // namespace c10
