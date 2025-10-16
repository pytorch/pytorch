#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/Stream.h>
#include <c10/core/thread_pool.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/llvmMathExtras.h>
#include <iostream>
#include <optional>

#include <deque>
#include <mutex>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
namespace at {

using c10::CachingAllocator::Stat;
using c10::CachingAllocator::DurationStat;

/**
 * HostBlock is typically a fundamental memory block used in pinned memory. It
 * is likely related to Event and Stream of device runtime. It is probably a
 * base struct or interface that can be inherited and extended by each backend.
 */
template <typename S>
struct HostBlock {
  // constructor for search key
  HostBlock(size_t size) : size_(size) {}

  HostBlock(size_t size, void* ptr) : size_(size), ptr_(ptr) {}

  std::mutex mutex_;
  size_t size_{0}; // block size in bytes
  void* ptr_{nullptr}; // memory address
  bool allocated_{false}; // in-use flag
  size_t event_count_{0}; // number of related events
  ska::flat_hash_set<S> streams_; // streams on which the block was used
};

template <typename B>
struct alignas(64) FreeBlockList {
  std::mutex mutex_;
  std::deque<B*> list_;
};

namespace {
  // Max cached block sizes: (1 << MAX_SIZE_INDEX) bytes
  // NOLINTNEXTLINE(misc-definitions-in-headers)
  constexpr size_t MAX_SIZE_INDEX = 64;
}

// A large reserved pinned memory segment that is created in advance which is used
// to allocate small pinned memory requests to avoid calling into expensive APIs.
// We never free this memory and move up the pointer as we allocate new blocks
// and when blocks are freed, they are cached in the free lists.
struct PinnedReserveSegment {
  PinnedReserveSegment(void *start, size_t size) : start_(start), size_(size),
    current_ptr_(start_), initialized_(true) {}

  PinnedReserveSegment() : start_(nullptr), size_(0), current_ptr_(nullptr), initialized_(false) {}

  bool initialized() {
    return initialized_;
  }

  void* allocate(size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);

    // Round up the requested size to 4KB boundary for all including the small ones.
    size_t rounded_bytes = (bytes + 4096 - 1) & ~(4096 - 1);

    if (((uint8_t*)current_ptr_ + rounded_bytes) > ((uint8_t*)start_ + size_)) {
      return nullptr;
    }

    void* ptr = current_ptr_;
    current_ptr_ = (uint8_t*)current_ptr_ + rounded_bytes;
    return ptr;
  }

  bool owns(void* ptr) {
    return ptr >= start_ && ptr < (uint8_t*)start_ + size_;
  }

  std::mutex mutex_;
  void* start_;
  size_t size_;
  void* current_ptr_;
  bool initialized_;
};

// Struct containing memory allocator summary statistics for host.
struct TORCH_API HostStats {
  // COUNT: total allocations (active)
  Stat active_requests;
  // SUM: bytes allocated/reserved by this memory alocator. (active)
  Stat active_bytes;
  // COUNT: total allocations (active + free)
  Stat allocations;
  // SUM: bytes allocated/reserved by this memory alocator. This accounts
  // for both free and in-use blocks.
  Stat allocated_bytes;

  // SUM: time spent in cudaHostAlloc/cudaHostRegister in microseconds
  DurationStat host_alloc_time;

  // SUM: time spent in cudaHostFree/cudaHostUnregister in microseconds
  DurationStat host_free_time;

  // COUNT: number of times cudaHostAlloc/cudaHostRegister was called because
  // the request could not be satisfied from existing free blocks.
  int64_t num_host_alloc = 0; // This is derived from segment or timing

  // COUNT: number of times cudaHostFree/cudaHostUnregister was called.
  int64_t num_host_free = 0; // This is derived from segment or timing

  // Count of cudaHostAlloc/cudaHostRegister per bucket
  std::vector<int64_t> bucket_allocation = std::vector<int64_t>(MAX_SIZE_INDEX);
};

// Struct containing memory allocator summary statistics for host, as they
// are staged for reporting. This is a temporary struct that is used to
// avoid locking the allocator while collecting stats.
struct alignas(64) HostStatsStaged {
  std::mutex timing_mutex_;
  // COUNT: total allocations (active + free)
  // LOCK: access to this stat is protected by the allocator's blocks_mutex_
  Stat allocations;
  // SUM: bytes allocated/reserved by this memory alocator. This accounts
  // for both free and in-use blocks.
  Stat allocated_bytes;
  // COUNT: number of allocations per bucket (active)
  // LOCK: access to this stat is protected by the per bucket free_list_[index].mutex_
  std::vector<Stat> active_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
  // SUM: bytes of allocation per bucket (active)
  // LOCK: access to this stat is protected by the per bucket free_list_[index].mutex_
  std::vector<Stat> active_bytes_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
  // COUNT: number of allocations per bucket (active + free)
  // LOCK: access to this stat is protected by the per bucket free_list_[index].mutex_
  std::vector<Stat> allocation_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
  // SUM: bytes of allocation per bucket (active + free)
  // LOCK: access to this stat is protected by the per bucket free_list_[index].mutex_
  std::vector<Stat> allocated_bytes_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
  // SUM: time spent in cudaHostAlloc/cudaHostRegister
  // LOCK: access to this stat is protected by the timing_mutex_
  DurationStat host_alloc_time;
  // SUM: time spent in cudaHostFree/cudaHostUnregister
  // LOCK: access to this stat is protected by the timing_mutex_
  DurationStat host_free_time;
};

/**
 * Note [HostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores runtime events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as CUDA runtime API calls)
 * while holding the lock.
 *
 * There are four public methods: allocate, free, record_event and empty_cache.
 *   1) In the allocate path, we first check to see if we can service our
 * request from this free list, and otherwise we create a new block with
 * allocate_host_memory.
 *   2) In the free path, we insert events (if required) into the event queue,
 * and if possible insert our block back into the free list. In allocate, we
 * first eagerly query events until we find one that is not ready, and insert
 * the corresponding block onto the free list if all the events recorded for a
 * block are ready.
 *   3) In the record_event path, we simply insert the given stream into the set
 * of streams tracked by the specified block. This set of streams is then
 * consumed in the free path.
 *   4) In the empty_cache path, we flush any available blocks into the free
 * list. Remove all element of free list, then remove them from block list and
 * release the associated pinned memory allocation via free_block.
 *
 * We generalize the caching host allocator into two parts: interface and
 * implementation. For any new backend looking to integrate with host allocator
 * and reuse caching mechanism, these two parts are necessary to be specialized.
 *
 * For the implementation, we provide a CachingHostAllocatorImpl struct
 * to abstract the caching mechanism. Any backend needs to provide a customized
 * implementation by specializing its own public functions and the related
 * runtime functions. Its template parameter S represents runtime Stream, E
 * denotes runtime Event, B indicates the fundamental memory block.
 *
 * For the interface, we provide a CachingHostAllocatorInterface struct as an
 * interface. Any backend needs to derive its own host allocator from this
 * interface. Its template parameter T refers to an implementation that
 * inherited from CachingHostAllocatorImpl.
 *
 * So this design can share the caching mechanism across each backend, and
 * provide flexibility to each backend. A backend can choose to follow this
 * implementation or reuse them by extending and overriding them as necessary.
 * Taking CUDA as an example, it specializes runtime related functions to reuse
 * the caching mechanism. Additionally, it extends the allocator's functionality
 * by adding the allocWithCudaHostRegister function to support page-locking the
 * memory range used by CUDA. Of course, you can also refer to
 * XPUCachingHostAllocator, which is a host caching allocator supported on XPU
 * backend, to implement a basic host caching allocator.
 *
 * Some of the invariants here are less strict than they could be - for example,
 * we do not enforce that free(Block* block) => block->event_count == 0. This is
 * for compatibility reasons, and we can explore enforcing these in subsequent
 * versions.
 *
 * Note that this caching host allocator does not split larger allocations into
 * smaller blocks, unlike the caching device allocator.
 *
 * In order to gather statistics about caching host allocator while minimally
 * impacting performance, we use a HostStatsStaged struct to stage the stats
 * before reporting them. This is done to avoid adding new locks to the allocator.
 * Collecting stats is carefully done under existing locks, and then the staged
 * stats are converted to the final stats when getStats is called. At that time
 * we hold the same locks as empty_cache, to ensure the fidelity of the stats.
 */

template <
    typename S,
    typename E,
    typename B = HostBlock<S>>
struct CachingHostAllocatorImpl {
  virtual ~CachingHostAllocatorImpl() {
    active_ = false;
    if (pinned_use_background_threads()) {
      getBackgroundThreadPool()->waitWorkComplete();
    }
  }

 public:
  // return data_ptr and block pair.
  virtual std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    // If we are using background threads, we can process events in the
    // background.
    if (!pinned_use_background_threads()) {
      process_events();
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    // These power of two sizes are also used to index into the free list.
    size_t roundSize = c10::llvm::PowerOf2Ceil(size);

    // First, try to allocate from the free list
    auto* block = get_free_block(roundSize);
    if (block) {
      return {block->ptr_, reinterpret_cast<void*>(block)};
    }

    // Check in the recently freed blocks with pending events to see if we
    // can reuse them. Call get_free_block again after processing events
    if (pinned_use_background_threads()) {
      // Launch the background thread and process events in a loop.
      static bool background_thread_flag [[maybe_unused]] = [this] {
        getBackgroundThreadPool()->run([&]() {
          while (active_) {
            process_events();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        });
        return true;
      }();
    }

    // Slow path: if we can't allocate from the cached free list, we need
    // to create a new block.
    void* ptr = nullptr;
    allocate_host_memory(roundSize, &ptr);

    // Then, create a new block.
    block = new B(roundSize, ptr);
    block->allocated_ = true;

    add_allocated_block(block);
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  virtual void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc, and thus we
    // do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<B*>(ctx);

    std::optional<std::vector<E>> events;
    ska::flat_hash_set<S> streams;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<E>();
        events->reserve(block->streams_.size());
        block->event_count_ += block->streams_.size();
        // Move out streams to avoid holding the mutex during event recording
        streams = std::move(block->streams_);
        block->streams_.clear();
      }
    }

    // Event recording must be done outside the mutex to avoid potential
    // deadlocks (e.g., when Python GIL is involved)
    for (auto stream : streams) {
      record_stream(events, stream);
    }

    if (!events) {
      auto index = size_index(block->size_);
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      free_list_[index].list_.push_back(block);
    } else {
      // restore these events that record by used streams.
      std::lock_guard<std::mutex> g(events_mutex_);
      for (auto&& event : *events) {
        events_.emplace_front(std::move(event), block);
      }
    }
  }

  virtual bool record_event(void* ptr, void* ctx, c10::Stream s) {
    S stream = S(s);
    auto* block = reinterpret_cast<B*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  virtual void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutexes and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      std::vector<B*> blocks_to_remove(free_list_[i].list_.begin(), free_list_[i].list_.end());
      free_list_[i].list_.clear();

      for (auto* block : blocks_to_remove) {
        blocks_.erase(block);
        ptr_to_block_.erase(block->ptr_);
        auto index = size_index(block->size_);
        free_block(block);
        stats_.allocations.decrease(1);
        stats_.allocated_bytes.decrease(block->size_);
        stats_.allocation_bucket_stats[index].decrease(1);
        stats_.allocated_bytes_bucket_stats[index].decrease(block->size_);
        delete block;
      }
    }
  }

  inline size_t size_index(size_t size) {
    return c10::llvm::Log2_64_Ceil(size);
  }

  virtual bool pinned_use_background_threads() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        pinned_use_background_threads();
  }

  virtual void copy_data(void* dest [[maybe_unused]], const void* src [[maybe_unused]], std::size_t count [[maybe_unused]]) const {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");
  }

  HostStats getStats() {
    HostStats stats;

    // To keep getStats lightweight we do *not* flush any available blocks
    // into the free_list. This may skew the stats a bit.

    auto add_bucket_stats = [](Stat& accumulator, const Stat& other) {
      accumulator.allocated += other.allocated;
      accumulator.current += other.current;
      accumulator.freed += other.freed;
      // Since peaks are measured per bucket independently, we add them up
      // to estimate the total peak. This is not strictly correct, but it is
      // the best approximation we can get after the fact.
      accumulator.peak += other.peak;
    };

    // Accurate reading of memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      // We collect the slow-path stats only once, since they are not collected
      // per bucket (we pick index 0 arbitrarily). These are also all the host
      // allocations, not taking into account caching and free lists.
      if (i == 0) {
        stats.allocations = stats_.allocations;
        stats.allocated_bytes = stats_.allocated_bytes;
        stats.num_host_alloc = stats.allocations.allocated;
        stats.num_host_free = stats.allocations.freed;
      }

      // Bucket stats need to be merged with the slow-path stats. We do this in
      // a best effort manner, since we can't really replay the cached events per bucket.
      add_bucket_stats(stats.active_requests, stats_.active_bucket_stats[i]);
      add_bucket_stats(stats.active_bytes, stats_.active_bytes_bucket_stats[i]);
      stats.bucket_allocation[i] = stats_.allocation_bucket_stats[i].allocated;
    }

    // Get the timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);

      stats.host_alloc_time = stats_.host_alloc_time;
      stats.host_free_time = stats_.host_free_time;
    }

    return stats;
  }

  void resetAccumulatedStats() {
    // Reseting accumulated memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      if (i == 0) {
        stats_.allocations.reset_accumulated();
        stats_.allocated_bytes.reset_accumulated();
      }
      stats_.active_bucket_stats[i].reset_accumulated();
      stats_.active_bytes_bucket_stats[i].reset_accumulated();
      stats_.allocation_bucket_stats[i].reset_accumulated();
      stats_.allocated_bytes_bucket_stats[i].reset_accumulated();
    }

    // Also reset timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.reset_accumulated();
      stats_.host_free_time.reset_accumulated();
    }
  }

  void resetPeakStats() {
    // Reseting peak memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      if (i == 0) {
        stats_.allocations.reset_peak();
        stats_.allocated_bytes.reset_peak();
      }
      stats_.active_bucket_stats[i].reset_peak();
      stats_.active_bytes_bucket_stats[i].reset_peak();
      stats_.allocation_bucket_stats[i].reset_peak();
      stats_.allocated_bytes_bucket_stats[i].reset_peak();
    }

    // Also reset timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.reset_peak();
      stats_.host_free_time.reset_peak();
    }
  }

 private:
  virtual void add_allocated_block(B* block) {
    std::lock_guard<std::mutex> g(blocks_mutex_);
    blocks_.insert(block);
    stats_.allocations.increase(1);
    stats_.allocated_bytes.increase(block->size_);
    ptr_to_block_.insert({block->ptr_, block});

    // Unfortunately, we have to, on the slow path, quickly
    // lock the bucket to record the allocation. This should
    // be a rare event once the cache is warmed up.
    auto size = block->size_;
    auto index = size_index(size);
    {
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      stats_.allocation_bucket_stats[index].increase(1);
      stats_.allocated_bytes_bucket_stats[index].increase(size);
      stats_.active_bucket_stats[index].increase(1);
      stats_.active_bytes_bucket_stats[index].increase(size);
    }
  }

  virtual B* get_free_block(size_t size) {
    auto index = size_index(size);
    std::lock_guard<std::mutex> g(free_list_[index].mutex_);
    if (!free_list_[index].list_.empty()) {
      B* block = free_list_[index].list_.back();
      free_list_[index].list_.pop_back();
      block->allocated_ = true;
      stats_.active_bucket_stats[index].increase(1);
      stats_.active_bytes_bucket_stats[index].increase(size);
      return block;
    }
    return nullptr;
  }

  virtual void process_events() {
    // process all events until the last unready event, not for specific size.
    process_events_for_specific_size(-1);
  }

  // If size is -1, process all events from backwards until the last unready
  // event. Otherwise, process events for a specific size and on first ready block
  // is found, add it to the free list and return.
  virtual void process_events_for_specific_size(int64_t size) {
    size_t event_count = 0;
    size_t max_events = 0;
    {
      std::lock_guard<std::mutex> g(events_mutex_);
      max_events = events_.size();
    }

    while (true) {
      // Avoid calling cudaEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      // process the last event
      std::optional<std::pair<E, B*>> processed;
      {
        std::lock_guard<std::mutex> g(events_mutex_);
        if (!events_.empty()) {
          processed = std::move(events_.back());
          events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      if (size != -1) {
        if (event_count++ > max_events) {
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            events_.push_front(std::move(*processed));
          }
          return;
        }
        if (size != (int64_t)processed->second->size_) {
          // if we are processing a specific size, and the size of the block
          // doesn't match, we can't use it.
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            events_.push_front(std::move(*processed));
          }
          continue;
        }
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        if (!query_event(event)) {
          // push the event onto the back if it's not ready.
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            if (size == -1) {
              events_.push_back(std::move(*processed));
              return;
            } else {
              events_.push_front(std::move(*processed));
              continue;
            }
          }
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        auto index = size_index(block->size_);
        std::lock_guard<std::mutex> g(free_list_[index].mutex_);
        free_list_[index].list_.push_back(block);
        stats_.active_bucket_stats[index].decrease(1);
        stats_.active_bytes_bucket_stats[index].decrease(size);
        if (size != -1) {
          return;
        }
      }
    }
  }

  TaskThreadPool* getBackgroundThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(1);
    return pool;
  }

  /* These following functions are runtime-related. */

  // Allocate page-locked memory on the host.
  virtual void allocate_host_memory(size_t size, void** ptr) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "Not implemented for allocate_host_memory");
  }

  // Free block and release the pointer contained in block.
  virtual void free_block(B* block) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for free_block");
  }

  // Record an event on stream and store event into events.
  virtual void record_stream(std::optional<std::vector<E>>& events, S stream) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for record_stream");
  }

  // Query event if it is completed.
  virtual bool query_event(E& event) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for query_event");
  }

  alignas(64) std::mutex blocks_mutex_;
  ska::flat_hash_set<B*> blocks_; // block list
  ska::flat_hash_map<void*, B*> ptr_to_block_;

  // We keep free list as a vector of free lists, one for each power of two
  // size. This allows us to quickly find a free block of the right size.
  // We use deque to store per size free list and guard the list with its own
  // mutex.
  alignas(64) std::vector<FreeBlockList<B>> free_list_ =
      std::vector<FreeBlockList<B>>(MAX_SIZE_INDEX);

  alignas(64) std::mutex events_mutex_;
  std::deque<std::pair<E, B*>> events_; // event queue paired with block

  // Indicates whether the object is active.
  // Set to false in the destructor to signal background threads to stop.
  std::atomic<bool> active_{true};
protected:
  alignas(64) HostStatsStaged stats_;
};

struct TORCH_API HostAllocator : public at::Allocator {
  // Associates the pinned memory allocation with a stream to track
  // dependencies. This ensures the memory won't be reused until the stream's
  // operations complete
  virtual bool record_event(void* ptr, void* ctx, c10::Stream stream) = 0;

  // Frees all cached pinned memory and returns it to the system, clearing the
  // allocator's internal cache
  virtual void empty_cache() = 0;

  // Returns comprehensive statistics about the allocator's memory usage,
  // allocation patterns, and timing metrics
  virtual HostStats get_stats() = 0;

  // Resets the cumulative allocation statistics
  virtual void reset_accumulated_stats() = 0;

  // Resets the peak memory usage metrics
  virtual void reset_peak_stats() = 0;
};

template <typename T, c10::DeleterFnPtr deleteFunc>
struct CachingHostAllocatorInterface : public HostAllocator {
  CachingHostAllocatorInterface() : impl_(std::make_unique<T>()) {}

  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        deleteFunc, // Use the template parameter deleter function
        at::DeviceType::CPU};
  }

  void free(void* ctx) {
    impl_->free(ctx);
  }

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    return impl_->record_event(ptr, ctx, stream);
  }

  void empty_cache() override {
    impl_->empty_cache();
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    impl_->copy_data(dest, src, count);
  }

  HostStats get_stats() override {
    return impl_->getStats();
  }

  void reset_accumulated_stats() override {
    impl_->resetAccumulatedStats();
  }

  void reset_peak_stats() override {
    impl_->resetPeakStats();
  }

  std::unique_ptr<T> impl_;
};

#define DECLARE_HOST_ALLOCATOR(name, impl, deleter, instance)       \
  void deleter(void* ptr);                                          \
  struct name final                                                 \
      : public at::CachingHostAllocatorInterface<impl, deleter> {}; \
  static name instance;                                                    \
  void deleter(void* ptr) {                                         \
    instance.free(ptr);                                             \
  }

/**
 * Set the host allocator for DeviceType `device_type`. This allocator manages
 * pinned memory on the host that can be accessed efficiently by the specified
 * device type. Note that this function is not thread-safe.
 */
TORCH_API void setHostAllocator(
    at::DeviceType device_type,
    at::HostAllocator* allocator,
    uint8_t priority = 0);

TORCH_API at::HostAllocator* getHostAllocator(at::DeviceType device_type);

template <DeviceType device_type>
struct HostAllocatorRegistry {
  explicit HostAllocatorRegistry(HostAllocator* allocator) {
    at::setHostAllocator(device_type, allocator);
  }
};

#define REGISTER_HOST_ALLOCATOR(device_type, allocator) \
  namespace {                                           \
  static at::HostAllocatorRegistry<device_type>         \
      g_host_allocator_registry_instance(allocator);    \
  }

} // namespace at
C10_DIAGNOSTIC_POP()
