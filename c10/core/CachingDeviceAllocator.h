#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/static_tracepoint.h>

#include <mutex>
#include <new>
#include <set>

namespace c10::CachingDeviceAllocator {

using namespace c10::CachingAllocator;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory allocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to device malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to device memory allocation
  // after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

} // namespace c10::CachingDeviceAllocator

namespace c10 {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by Graph mode capture_begin.
// second is set if the instance is created by Graph mode graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

struct C10_API DeviceAllocator : public c10::Allocator {
  DeviceAllocator();
  ~DeviceAllocator() override;

  // Returns true if the allocator has been properly initialized and is ready
  // for use
  virtual bool initialized() = 0;

  // Releases all cached device memory from the specified memory pool back to
  // the system
  virtual void emptyCache(MempoolId_t mempool_id = {0, 0}) = 0;

  // Associates a memory allocation with a stream to establish dependency
  // tracking. Prevents memory reuse until all operations on the specified
  // stream complete
  virtual void recordStream(const DataPtr& ptr, c10::Stream stream) = 0;

  // Retrieves comprehensive memory statistics for the specified device,
  // including allocation patterns, usage metrics
  virtual CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) = 0;

  // Resets cumulative allocation statistics for the specified device to zero
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;

  // Resets peak memory usage statistics for the specified device
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;

  // Return the free memory size and total memory size in bytes for the
  // specified device.
  virtual std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) = 0;
};

// This function is used to get the DeviceAllocator for a specific device type
// and keep backward compatibility with c10::GetAllocator.
C10_API inline DeviceAllocator* getDeviceAllocator(const DeviceType& t) {
  TORCH_CHECK(
      t != DeviceType::CPU,
      "getDeviceAllocator is not supported for CPU device type.");
  auto* allocator = c10::GetAllocator(t);
  auto* device_allocator = dynamic_cast<DeviceAllocator*>(allocator);
  TORCH_INTERNAL_ASSERT(
      device_allocator, "Allocator for ", t, " is not a DeviceAllocator.");
  return device_allocator;
}

constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
const size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
static constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

template <
    typename ImplT,
    typename BlockT,
    c10::DeleterFnPtr deleteFunc,
    typename BaseDeviceAllocator = c10::DeviceAllocator>
struct CachingDeviceAllocatorInterface : public BaseDeviceAllocator {
  BlockT* get_allocated_block(void* ptr, bool remove = false) {
    const auto mutex_shard_id = get_mutex_shard_id(ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    auto it = allocated_blocks[mutex_shard_id].find(ptr);
    if (it == allocated_blocks[mutex_shard_id].end()) {
      return nullptr;
    }
    BlockT* block = it->second;
    if (remove) {
      allocated_blocks[mutex_shard_id].erase(it);
    }
    return block;
  }

  at::DataPtr allocate(size_t size) override {
    c10::impl::VirtualGuardImpl impl(ImplT::static_device_type);
    c10::Device device = impl.getDevice();
    void* devPtr = nullptr;
    c10::Stream stream = impl.getStream(device);
    if (size != 0) {
      this->malloc(&devPtr, device.index(), size, stream);
    }

    return {devPtr, devPtr, deleteFunc, device};
  }

  void malloc(
      void** devPtr,
      c10::DeviceIndex device,
      size_t size,
      c10::Stream stream) {
    checkDeviceIndex(device);
    BlockT* block = nullptr;
    block = impls_[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          ImplT::static_device_type, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    BlockT* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          ImplT::static_device_type, reinterpret_cast<uintptr_t>(block->ptr));
    }
    impls_[block->device]->free(block);
  }

  DeleterFnPtr raw_deleter() const override {
    return deleteFunc;
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when device tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != deleteFunc) {
      return;
    }

    BlockT* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_CHECK(block, "No allocated block can be found.");
    impls_[block->device]->recordStream(block, stream);
  }

  void empty_cache(MempoolId_t mempool_id = {0, 0}) override {
    for (auto& impl : impls_) {
      impl->emptyCache(mempool_id);
    }
  }

  CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override {
    return impls_[device]->getDeviceStats();
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    impls_[device]->resetAccumulatedStats();
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    impls_[device]->resetPeakStats();
  }

  // Initializes the per-device allocator implementations based on the given
  // device count. This method must be called before any allocator is used.
  void init(c10::DeviceIndex device_count) {
    const auto size = static_cast<c10::DeviceIndex>(impls_.size());
    if (size < device_count) {
      impls_.resize(device_count);
      for (const auto& i : c10::irange(size, device_count)) {
        impls_[i] = std::make_unique<ImplT>(i);
      }
    }
  }

  bool initialized() override {
    return !impls_.empty();
  }

 private:
  void checkDeviceIndex(DeviceIndex device_index) const {
    TORCH_CHECK(
        0 <= device_index && device_index < impls_.size(),
        "Invalid device argument ",
        static_cast<int>(device_index),
        ": did you call init?");
  }

  void add_allocated_block(BlockT* block) {
    const auto mutex_shard_id = get_mutex_shard_id(block->ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    allocated_blocks[mutex_shard_id][block->ptr] = block;
  }

  static size_t get_mutex_shard_id(void* ptr) {
    return twang_mix64(reinterpret_cast<size_t>(ptr)) % kNumMutexShard;
  }

  // Aligns the mutex to avoid false sharing on cache lines.
  struct alignas(hardware_destructive_interference_size) AlignedMutex {
    std::mutex m;
  };
  // A prime number close to 64, used for sharding to reduce contention.
  static constexpr size_t kNumMutexShard = 67;
  std::array<AlignedMutex, kNumMutexShard> mutex;
  // A map of allocated blocks, sharded by mutex to reduce contention.
  std::array<ska::flat_hash_map<void*, BlockT*>, kNumMutexShard>
      allocated_blocks{};
  // Per-device allocator implementations.
  std::vector<std::unique_ptr<ImplT>> impls_{};
};

/**
 * Note [DeviceAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */

template <typename BlockT>
struct BlockComparatorSize {
  bool operator()(const BlockT* a, const BlockT* b) const {
    if (a->size != b->size) {
      return a->size < b->size;
    }
    if (a->stream != b->stream) {
      return reinterpret_cast<uintptr_t>(a->stream) <
          reinterpret_cast<uintptr_t>(b->stream);
    }
    return reinterpret_cast<uintptr_t>(a->ptr) <
        reinterpret_cast<uintptr_t>(b->ptr);
  }
};

template <typename BlockT>
struct BlockComparatorAddress {
  bool operator()(const BlockT* a, const BlockT* b) const {
    if (a->stream != b->stream) {
      return reinterpret_cast<uintptr_t>(a->stream) <
          reinterpret_cast<uintptr_t>(b->stream);
    }
    return reinterpret_cast<uintptr_t>(a->ptr) <
        reinterpret_cast<uintptr_t>(b->ptr);
  }
};

// Forward declaration
template <typename BlockT>
struct PrivatePool;

template <typename BlockT>
struct BlockPool {
  BlockPool(bool small, PrivatePool<BlockT>* private_pool = nullptr)
      : blocks(BlockComparatorSize<BlockT>()),
        unmapped(BlockComparatorAddress<BlockT>()),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<BlockT*, BlockComparatorSize<BlockT>> blocks{};
  std::set<BlockT*, BlockComparatorAddress<BlockT>> unmapped{};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
  PrivatePool<BlockT>* owner_PrivatePool;
  int64_t get_free_blocks_call_count{0};

  // Add a Block into blocks set with updating gc counter.
  std::pair<
      typename std::set<BlockT*, BlockComparatorSize<BlockT>>::iterator,
      bool>
  insert_into_blocks(BlockT* block) {
    block->gc_count_base = get_free_blocks_call_count;
    return blocks.insert(block);
  }

  MempoolId_t owner_MempoolId() const {
    if (owner_PrivatePool) {
      return owner_PrivatePool->id;
    } else {
      return {0, 0};
    }
  }
};

template <typename StreamT, typename HandleT>
struct ExpandableSegment;

template <typename StreamT, typename HandleT = void*>
struct DeviceBlock {
  using BlockT = DeviceBlock<StreamT>;
  using BlockPoolT = BlockPool<BlockT>;

  DeviceBlock(
      c10::DeviceIndex device,
      StreamT stream,
      size_t size,
      BlockPoolT* pool,
      void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  DeviceBlock(c10::DeviceIndex device, StreamT stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  size_t gc_count() const {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  void splice(BlockT* before, BlockT* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }

  c10::DeviceIndex device; // gpu
  StreamT stream; // allocation stream
  ska::flat_hash_set<StreamT>
      stream_uses{}; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPoolT* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // is the virtual address range this Block references
  // Backed by physical pages. Always true when
  // expandable_segment_ is null. When false
  // This Block will be aligned to the segment size
  // of its expandable_segment_.
  BlockT* prev{nullptr}; // prev block if split from a larger allocation
  BlockT* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int64_t gc_count_base{0}; // get_free_blocks_call_count when Block is inserted
  std::shared_ptr<GatheredContext> context_when_allocated;
  // Only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;
  // Expandble segment this block belongs to.
  ExpandableSegment<StreamT, HandleT>* expandable_segment_{nullptr};
};

template <typename BlockT>
struct PrivatePool {
  using BlockPoolT = BlockPool<BlockT>;
  PrivatePool(MempoolId_t id, DeviceAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  C10_DISABLE_COPY_AND_ASSIGN(PrivatePool);
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed device allocation made for this pool. When use_count and
  // deviceMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int deviceMalloc_count{0};
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critical though,
  // I'd rather not add more logic to it.
  DeviceAllocator* allocator_;
  BlockPoolT large_blocks;
  BlockPoolT small_blocks;

 public:
  DeviceAllocator* allocator() {
    return allocator_;
  }
};

// Represents a contiguous virtual memory segment mapped for allocation.
struct SegmentRange {
  SegmentRange(void* p, size_t s) : ptr_(static_cast<char*>(p)), size_(s) {}

  char* ptr_; // Starting address of the mapped range.
  size_t size_; // Size in bytes of the mapped range.
};

template <typename StreamT, typename HandleT = void*>
struct ExpandableSegment {
  ExpandableSegment() = default;
  C10_DISABLE_COPY_AND_ASSIGN(ExpandableSegment);
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment& operator=(ExpandableSegment&&) = delete;

  virtual void init(
      c10::DeviceIndex device,
      std::optional<StreamT> stream,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers) {
    device_ = device;
    stream_ = std::move(stream);
    // 2MB for small pool, 20MB for large pool
    segment_size_ = segment_size;
    peers_ = std::move(peers);
    max_handles_ = numSegments(getReservedVirtualMemorySize());
    createVirtualMemoryAddress(&ptr_);
  }

  // Maps a virtual memory range to physical memory.
  virtual SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr_);
    auto end = segmentRight(range.ptr_ + range.size_);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr_);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    mapHandles(begin, end);
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
    return rangeFromHandles(begin, end);
  }

  // Unmap a virtual memory range from physical memory.
  virtual SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr_);
    auto end = segmentLeft(range.ptr_ + range.size_);
    if (begin >= end) {
      return SegmentRange{range.ptr_, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<char*>(ptr_);
  }

  size_t size() const {
    return max_handles_ * segment_size_;
  }

  // Registers a new peer device and updates access permissions.
  virtual void addPeer(c10::DeviceIndex device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  virtual ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    releaseVirtualMemoryAddress(ptr_);
  }

 private:
  // Runtime-related methods

  // Returns the reserved virtual memory size for this segment, which may be
  // larger than the total size if the segment is expandable.
  virtual size_t getReservedVirtualMemorySize() = 0;

  // Create virtual memory address for this segment for the reserved size.
  virtual void createVirtualMemoryAddress(void** ptr) = 0;

  // Release the virtual memory address associated with the segment.
  virtual void releaseVirtualMemoryAddress(void* ptr) = 0;

  // Maps the physical memory handles in the range [begin, end) to the segment.
  virtual void mapHandles(size_t begin, size_t end) = 0;

  // Unmaps the physical memory handles in the range [begin, end) from the
  // segment.
  virtual void unmapHandles(size_t begin, size_t end) = 0;

  // Sets access permissions for the specified device on the segment range
  // [begin, end).
  virtual void setAccess(c10::DeviceIndex device, size_t begin, size_t end) = 0;

  // Internal methods

  // Returns the number of full segments required to cover `size` bytes.
  // Rounds up to ensure partial segments are counted.
  size_t numSegments(size_t size) const {
    return (size + segment_size_ - 1) / segment_size_;
  }

  // Returns the index of the segment that contains the pointer `p`,
  // relative to the base pointer `ptr_`. This is the *inclusive* lower bound
  // of the segment that includes `p`.
  size_t segmentLeft(char* p) const {
    size_t offset = p - ptr();
    return offset / segment_size_;
  }

  // Returns the index of the segment just *past* the one containing pointer
  // `p`, relative to the base pointer `ptr_`. This is the *exclusive* upper
  // bound, useful for [begin, end) style ranges.
  // If `p` lies exactly on a segment boundary, this is equal to segmentLeft(p).
  // Otherwise, it rounds up and returns segmentLeft(p) + 1.
  size_t segmentRight(char* p) const {
    size_t offset = p - ptr();
    return numSegments(offset);
  }

  // Constructs a SegmentRange starting at [start, end) indices.
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  // Iterates over all contiguous ranges of allocated segments in `handles_`,
  // and invokes the provided function `fn(start, end)` for each range.
  // Each range is defined as a half-open interval [start, end).
  void forEachAllocatedRange(const std::function<void(size_t, size_t)>& fn) {
    size_t start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  c10::DeviceIndex device_{-1};
  std::optional<StreamT> stream_;
  // Virtual memory address used in reserveVirtualMemory.
  void* ptr_{nullptr};
  // Size of each segment in bytes.
  size_t segment_size_{0};
  // Maximum number of segments that can be allocated in this segment.
  size_t max_handles_{0};
  // Physical memory handles for the segments.
  std::vector<std::optional<HandleT>> handles_{};
  // Peer devices on which this memory should be mapped and accessible.
  std::vector<c10::DeviceIndex> peers_{};
};

template <typename ExpandableSegmentT>
ExpandableSegmentT* make_expandable_segment(
    c10::DeviceIndex device,
    std::optional<typename ExpandableSegmentT::StreamT> stream,
    size_t segment_size,
    std::vector<c10::DeviceIndex> peers) {
  static_assert(
      std::is_base_of_v<
          ExpandableSegment<
              typename ExpandableSegmentT::StreamT,
              typename ExpandableSegmentT::HandleT>,
          ExpandableSegmentT>,
      "ExpandableSegmentT must inherit from ExpandableSegment<StreamT, HandleT>");
  ExpandableSegmentT* ptr = new ExpandableSegmentT();
  TORCH_INTERNAL_ASSERT(ptr, "Failed to allocate memory for ExpandableSegment");
  ptr->init(device, std::move(stream), segment_size, std::move(peers));
  return ptr;
}

template <typename StreamT, typename HandleT = void*>
struct AllocParams {
  using BlockT = DeviceBlock<StreamT, HandleT>;
  using BlockPoolT = BlockPool<BlockT>;

  enum class Status : uint8_t { Ok, OOM, Error };

  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      StreamT stream,
      BlockPoolT* pool,
      size_t alloc_size)
      : alloc_size(alloc_size), search_key(device, stream, size), pool(pool) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }
  StreamT stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  size_t alloc_size;
  BlockT search_key;
  // The block pool this allocation belongs to.
  BlockPoolT* pool;
  // This is the block that was allocated for this request.
  BlockT* block{nullptr};
  // Tracks which stats to update for this allocation.
  CachingAllocator::StatTypes stat_types{false};
  // Result status of the allocation attempt.
  Status status{Status::Ok};
};

template <
    typename StreamT,
    typename EventT,
    typename HandleT = void*,
    typename BlockT = DeviceBlock<StreamT, HandleT>,
    typename ES = ExpandableSegment<StreamT, HandleT>>
struct CachingDeviceAllocatorImpl {
  virtual ~CachingDeviceAllocatorImpl() = default;

  BlockT* malloc(c10::DeviceIndex device, size_t size, c10::Stream stream) {}

 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  CachingDeviceAllocator::DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool<BlockT> large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool<BlockT> small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<BlockT*> active_blocks;

  // captures_underway tracks if we are diverting some
  // allocations to a specific pool.
  // Most of the time it's empty, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  std::vector<std::pair<MempoolId_t, std::function<bool(StreamT)>>>
      captures_underway;

  // tracks which pools we can use as a last resort before ooming
  ska::flat_hash_set<MempoolId_t, MempoolIdHash> use_on_oom_pools;

  // Deferring event recording on blocks until Graph Capture Mode has completed.
  std::vector<BlockT*> needs_events_deferred_until_no_capture;
  
  // outstanding cuda events
  ska::flat_hash_map<
      StreamT,
      std::deque<std::pair<EventT, BlockT*>>>
      block_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment<StreamT, HandleT>*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  bool set_fraction = false;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  RecordContext record_context_ = RecordContext::NEVER;

  // Ring buffer for memory snapshot TraceEntry's
  RingBuffer<TraceEntry> alloc_buffer;

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool<BlockT>>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool<BlockT>*, MempoolIdHash>
      graph_pools_freeable;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

  std::vector<AllocatorTraceTracker> trace_trackers_;

  // mapping from block to a stream_set, containing streams on which the block
  // was used while cudagraph capturing
  std::unordered_map<BlockT*, ska::flat_hash_set<StreamT>> block_to_cudagraph_stream_uses;

  // thread local compile context for each device
  static thread_local std::stack<std::string> compile_context;

 public:
 private:
 protected:
};

} // namespace c10
