#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

#include <cuda_runtime_api.h>
#include <future>
#include <unordered_map>

namespace at::cuda {

template <typename T, c10::DeleterFnPtr deleteFunc>
struct CachingHostAllocatorInterface : public HostAllocator {
  CachingHostAllocatorInterface() : impl_(std::make_shared<T>()) {}

  using Block = typename T::Block;

  at::DataPtr allocate(size_t size) override {
    T* impl{};
    {
      std::shared_lock lock(mutex_);
      impl = getImplByFilter(getCurrentCUDAStream());
    }
    auto ptr_and_ctx = impl->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        deleteFunc, // Use the template parameter deleter function
        at::DeviceType::CPU};
  }

  void free(void* ctx) {
    TORCH_INTERNAL_ASSERT(ctx);
    auto* block = reinterpret_cast<Block*>(ctx);
    // We don't need to take the block's mutex for block->pool_ptr_
    // because pool_ptr_ never changes.
    block->pool_ptr_->free(ctx);
  }

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    // Warning: Sometimes a non-block ctx is passed to record event
    // Figure out what to do then.
    T* impl{};
    {
      std::shared_lock lock(mutex_);
      impl = getImplByFilter(getCurrentCUDAStream());
    }
    return impl->record_event(ptr, ctx, stream);
  }

  void create_or_incref_pool(c10::MempoolId_t pool_id) override {
    std::unique_lock lock(mutex_);
    createImplOrIncRefIfDoesNotExist(pool_id);
  }

  void begin_allocate_to_pool(
      c10::MempoolId_t pool_id,
      std::function<bool(c10::Stream)> filter) override {
    std::unique_lock lock(mutex_);
    for (auto it = captures_underway_.begin(); it != captures_underway_.end();
         ++it) {
      TORCH_CHECK(
          it->first != pool_id,
          "beginAllocateToPool: already recording to pool_id");
    }
    captures_underway_.emplace_back(pool_id, std::move(filter));
    createImplOrIncRefIfDoesNotExist(pool_id);
  }

  void end_allocate_to_pool(c10::MempoolId_t pool_id) override {
    std::unique_lock lock(mutex_);
    auto it = std::find_if(captures_underway_.begin(), captures_underway_.end(),
        [pool_id](const auto& elem) {
            return std::get<0>(elem) == pool_id;
        });
    TORCH_INTERNAL_ASSERT(it != captures_underway_.end());
    captures_underway_.erase(it);
  }

  void release_pool(c10::MempoolId_t pool_id) override {
    std::unique_lock lock(mutex_);
    auto it = private_impls_.find(pool_id);
    TORCH_INTERNAL_ASSERT(it != private_impls_.end());
    it->second.pop_back();
    if (it->second.empty()) {
      private_impls_.erase(it);
    }
  }

  // TODO: We should add support for empty_cache on a specific memory
  // pool, like CUDACachingAllocator has, at some point.
  void empty_cache(/*c10::MempoolId_t mempool_id = {0, 0}*/) override {
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

 private:
  T* getImplByFilter(c10::cuda::CUDAStream stream) {
    // iterate in reverse order to enforce a LIFO ordering when two
    // nested memory pools have filters that return true. See
    // https://github.com/pytorch/pytorch/issues/161193
    for (auto it = captures_underway_.rbegin(); it != captures_underway_.rend(); ++it) {
      auto &&[mempool_id, filter] = *it;
      if (filter(stream)) {
        return private_impls_.at(mempool_id).front().get();
      }
    }
    return impl_.get();
  }

  T* getImpl(c10::MempoolId_t pool_id) {
    if (pool_id == c10::MempoolId_t{0, 0}) {
      return impl_.get();
    }
    auto it = private_impls_.find(pool_id);
    TORCH_INTERNAL_ASSERT(it != private_impls_.end());
    return it->second.front().get();
  }

  void createImplOrIncRefIfDoesNotExist(c10::MempoolId_t pool_id) {
    TORCH_INTERNAL_ASSERT(pool_id != (c10::MempoolId_t{0, 0}));
    auto it = private_impls_.find(pool_id);
    if (it == private_impls_.end()) {
      private_impls_[pool_id].emplace_back(std::make_shared<T>());
    } else {
      private_impls_.at(pool_id).emplace_back(private_impls_.at(pool_id).front());
    }
  }

  ska::flat_hash_map<
      c10::MempoolId_t,
      std::vector<std::shared_ptr<T>>,
      c10::MempoolIdHash>
      private_impls_;

  std::vector<std::pair<MempoolId_t, std::function<bool(c10::Stream)>>> captures_underway_;

  std::shared_ptr<T> impl_;

  std::shared_mutex mutex_;
};

namespace {

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

template <
    typename S,
    typename E,
    typename B,
    typename Child>
struct CachingHostAllocatorImpl: std::enable_shared_from_this<Child> {
 private:
    Child& self() {
        return *static_cast<Child*>(this);
    }
    const Child& self() const {
        return *static_cast<const Child*>(this);
    }

 public:
  explicit CachingHostAllocatorImpl() {}

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
      block->allocating_stream_ = at::cuda::getCurrentCUDAStream();
      block->was_allocated_during_stream_capture_ =
        (at::cuda::currentStreamCaptureStatusMayInitCtx() !=
         at::cuda::CaptureStatus::None);
      return {block->ptr_, reinterpret_cast<void*>(block)};
    }

    // Check in the recently freed blocks with pending events to see if we
    // can reuse them. Call get_free_block again after processing events
    if (pinned_use_background_threads()) {
      process_events_for_specific_size(roundSize);
      block = get_free_block(roundSize);
      if (block) {
        block->allocating_stream_ = at::cuda::getCurrentCUDAStream();
        block->was_allocated_during_stream_capture_ =
          (at::cuda::currentStreamCaptureStatusMayInitCtx() !=
           at::cuda::CaptureStatus::None);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }

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
    block = new B(roundSize, ptr, self().shared_from_this(), at::cuda::getCurrentCUDAStream());
    block->allocated_ = true;

    add_allocated_block(block);
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  virtual void free(void* ctx) {
    if (!ctx) {
      return;
    }

    auto* block = reinterpret_cast<B*>(ctx);

    std::optional<std::vector<EventPool::Event>> events;
    ska::flat_hash_set<CUDAStream> streams;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        block->event_count_ += block->streams_.size();
        streams = std::move(block->streams_);
        block->streams_.clear();
      }

      if (streams.size() == 1 &&
          streams.count(block->allocating_stream_) &&
          block->was_allocated_during_stream_capture_) {
        // unless a block was used on only the stream it was allocated
        // on, it cannot ever be recycled during stream capture, until
        // we implement a more sophisticated algorithm.
        block->event_count_--;
        auto index = size_index(block->size_);
        std::lock_guard<std::mutex> g(free_list_[index].mutex_);
        free_list_[index].list_.push_back(block);
        stats_.allocation_bucket_stats[index].decrease(1);
        stats_.allocated_bytes_bucket_stats[index].decrease(block->size_);
        return;
      }
    }

    if (!block->was_allocated_during_stream_capture_) {
      for (auto stream : streams) {
        record_stream(events, stream);
      }
    }

    if (!events) {
      auto index = size_index(block->size_);
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      free_list_[index].list_.push_back(block);
      stats_.allocation_bucket_stats[index].decrease(1);
      stats_.allocated_bytes_bucket_stats[index].decrease(block->size_);
    } else {
      TORCH_INTERNAL_ASSERT(!block->was_allocated_during_stream_capture_ || events->empty());
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
        stats_.allocation.decrease(1);
        stats_.allocated_bytes.decrease(block->size_);
        free_block(block);
        delete block;
      }
    }
  }

  inline size_t size_index(size_t size) {
    return c10::llvm::Log2_64_Ceil(size);
  }

  virtual bool pinned_use_background_threads() {
    return false;
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
        stats.segment = stats_.allocation;
        stats.reserved_bytes = stats_.allocated_bytes;
        stats.num_host_alloc = stats.segment.allocated;
        stats.num_host_free = stats.segment.freed;
      }

      // Bucket stats need to be merged with the slow-path stats. We do this in
      // a best effort manner, since we can't really replay the cached events per bucket.
      add_bucket_stats(stats.allocation, stats_.allocation_bucket_stats[i]);
      add_bucket_stats(stats.allocated_bytes, stats_.allocated_bytes_bucket_stats[i]);
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
        stats_.allocation.reset_accumulated();
        stats_.allocated_bytes.reset_accumulated();
      }
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
        stats_.allocation.reset_peak();
        stats_.allocated_bytes.reset_peak();
      }
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
    stats_.allocation.increase(1);
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
    }
  }

  virtual B* get_free_block(size_t size) {
    auto index = size_index(size);
    std::lock_guard<std::mutex> g(free_list_[index].mutex_);
    if (!free_list_[index].list_.empty()) {
      B* block = free_list_[index].list_.back();
      free_list_[index].list_.pop_back();
      block->allocated_ = true;
      stats_.allocation_bucket_stats[index].increase(1);
      stats_.allocated_bytes_bucket_stats[index].increase(size);
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
        stats_.allocation_bucket_stats[index].decrease(1);
        stats_.allocated_bytes_bucket_stats[index].decrease(size);
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

 protected:
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
  alignas(64) HostStatsStaged stats_;
};

// forward declare
struct CUDACachingHostAllocatorImpl;

struct CUDACachingHostAllocatorImplBlock {
  CUDACachingHostAllocatorImplBlock(size_t size, void* ptr, std::shared_ptr<CUDACachingHostAllocatorImpl> pool_ptr, CUDAStream allocating_stream) : size_(size), ptr_(ptr), allocating_stream_(std::move(allocating_stream)), was_allocated_during_stream_capture_(at::cuda::currentStreamCaptureStatusMayInitCtx() != at::cuda::CaptureStatus::None), pool_ptr_(std::move(pool_ptr)) {}

  std::mutex mutex_;
  size_t size_{0}; // block size in bytes
  void* ptr_{nullptr}; // memory address
  bool allocated_{false}; // in-use flag
  size_t event_count_{0}; // number of related events
  ska::flat_hash_set<CUDAStream> streams_; // streams on which the block was used
  CUDAStream allocating_stream_; // stream which allocated the block
  bool was_allocated_during_stream_capture_;
  std::shared_ptr<CUDACachingHostAllocatorImpl> pool_ptr_;
};

struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, EventPool::Event, CUDACachingHostAllocatorImplBlock, CUDACachingHostAllocatorImpl> {
  using Block = CUDACachingHostAllocatorImplBlock;

  explicit CUDACachingHostAllocatorImpl(): CachingHostAllocatorImpl<CUDAStream, EventPool::Event, CUDACachingHostAllocatorImplBlock, CUDACachingHostAllocatorImpl>() {}

 private:
  std::unordered_map<void*, bool> use_host_register;

  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    auto start = std::chrono::steady_clock::now();
    bool use_register = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_host_register();
    if (use_register) {
      allocWithCudaHostRegister(ptr, size);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(*ptr) == 0);
      use_host_register[*ptr] = use_register;
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    auto start = std::chrono::steady_clock::now();
    // Users may change the allocator config at will. torch unit tests do this.
    // However, allocations using cudaHostRegister should use corresonding
    // cudaHostUnregister and similarly for cudaHostAlloc / cudaFreeHost.
    void* ptr = block->ptr_;
    bool use_register = false;
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(ptr) == 1);
      use_register = use_host_register[ptr];
    }
    if (use_register) {
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(ptr));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_host_register.erase(ptr);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    // It is rare, but query_event() can be called during stream
    // capture. This happens when stream capture is immediately
    // preceded by allocating to this pool via
    // _use_cuda_memory_pool_manager. Since capture_begin() is
    // preceded by torch.cuda.synchronize() in torch/cuda/graphs.py,
    // we can be certain that cudaEventQuery always returns
    // cudaSuccess in this rare situation.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    cudaError_t err = cudaEventQuery(*event);
    if (err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err != cudaSuccess) {
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  bool pinned_use_background_threads() override {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        pinned_use_background_threads();
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_max_register_threads()));
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      memset((void*)p, 0, 1);
    }
  }

  void allocWithCudaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    *ptr = std::malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using cudaHostRegister
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    AT_CUDA_CHECK(
        cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
  }
};

} // anonymous namespace


void raw_local_deleter(void* ptr);
struct CUDACachingHostAllocator final
  : public at::cuda::CachingHostAllocatorInterface<CUDACachingHostAllocatorImpl, raw_local_deleter> {};
static CUDACachingHostAllocator caching_host_allocator;
void raw_local_deleter(void* ptr) {
  caching_host_allocator.free(ptr);
}

REGISTER_HOST_ALLOCATOR(at::kCUDA, &caching_host_allocator)
} // namespace at::cuda
