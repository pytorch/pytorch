#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/static_tracepoint.h>

#include <deque>
#include <mutex>
#include <new>
#include <set>
#include <stack>

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

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    size_t allocated,
    size_t device_total,
    size_t device_free)>;

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

typedef std::shared_ptr<GatheredContext> (*CreateContextFn)();

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

union trace_time_ {
  time_t t_;
  approx_time_t approx_t_;
};

template <typename StreamT>
struct TraceEntry {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via record_stream
                    // This event is generated when a free actually completes.
    SEGMENT_ALLOC, // a call to device allocation like `cudaMalloc` to get more
                   // memory from the OS
    SEGMENT_FREE, // a call to device deallocation like `cudaFree` to return
                  // memory to the OS (e.g. to defragment or empty_caches)
    SEGMENT_MAP, // a call to virtual memory map like `cuMemMap` (used with
                 // expandable_segments)
    SEGMENT_UNMAP, // a call to virtual memory unmap like `cuMemUnmap` (used
                   // with expandable segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by cuda)
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      size_t addr,
      size_t size,
      StreamT stream,
      MempoolId_t mempool,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr,
      std::string compile_context = "")
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size),
        mempool_(std::move(mempool)),
        compile_context_(std::move(compile_context)) {
    time_.approx_t_ = time;
  }
  Action action_;
  c10::DeviceIndex device_;
  size_t addr_; // for OOM, this is the amount of free bytes reported by cuda
  std::shared_ptr<GatheredContext> context_;
  StreamT stream_{};
  size_t size_;
  MempoolId_t mempool_;
  trace_time_ time_{};
  std::string compile_context_{};
};

using AllocatorTraceTracker = std::function<void(const TraceEntry<StreamT>&)>;

template <class T>
class RingBuffer {
 public:
  RingBuffer() {
    // alloc_trace is a pointer because we need to intentionally
    // leak this on deallocation it can hold references to Python
    // state which will already be destroyed when we are in exit handlers
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    alloc_trace = new std::vector<T>();
  }

  void setMaxEntries(size_t size) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_max_entries_ = std::max(size_t(1), size);
  }

  void insertEntries(const T& entry) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(entry);
    } else {
      (*alloc_trace)[alloc_trace_next++] = entry;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }

  void getEntries(std::vector<T>& result) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() +
            static_cast<typename std::vector<T>::difference_type>(
                alloc_trace_next),
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() +
            static_cast<typename std::vector<T>::difference_type>(
                alloc_trace_next));
  }

  void clear() {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

 private:
  size_t alloc_trace_max_entries_ = 1;

  // Both alloc_trace and alloc_trace_next needs to be used
  // under alloc_trace_lock.
  std::mutex alloc_trace_lock;
  size_t alloc_trace_next = 0;
  std::vector<T>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers
};

template <
    typename StreamT,
    typename EventT,
    typename HandleT = void*,
    typename BlockT = DeviceBlock<StreamT, HandleT>,
    typename ES = ExpandableSegment<StreamT, HandleT>>
struct CachingDeviceAllocatorImpl {
  virtual ~CachingDeviceAllocatorImpl() = default;

  CachingDeviceAllocatorImpl(c10::DeviceIndex device_index)
      : device_index_(device_index), large_blocks(/*small=*/false), small_blocks(/*small=*/true) {
    stats.max_split_size =
        static_cast<int64_t>(AcceleratorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

  BlockT* malloc(c10::DeviceIndex device, size_t orig_size, c10::Stream c10_stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.size() > 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events(context);
    }
    StreamT stream = StreamT(c10_stream);
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
  }

   /** Returns a copy of the memory allocator stats **/
   DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

 private:

  /* internal functions */

  BlockPool<BlockT>& get_pool(size_t size, StreamT stream) {
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only non-empty if some thread has begun and not yet ended
    // a capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        if (entry.second(stream)) {
          auto it1 = graph_pools.find(entry.first);
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          if (size <= kSmallSize) {
            return it1->second->small_blocks;
          } else {
            return it1->second->large_blocks;
          }
        }
      }
    }
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 division.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

    TORCH_CHECK(divisions >= 2, "Only 2 or more divisions are supported");

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }
  
  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = AcceleratorConfig::roundup_power2_divisions(size);
      if (divisions > 1 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

  // lock around all operations
  mutable std::recursive_mutex mutex;

  c10::DeviceIndex device_index_;

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
  ska::flat_hash_map<StreamT, std::deque<std::pair<EventT, BlockT*>>>
      block_events;

  // Tracks memory usage during garbage collection.
  size_t total_allocated_memory = 0;

  // Upper limit of memory allowed during garbage collection.
  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment<StreamT, HandleT>*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  bool set_fraction = false;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  RecordContext record_context_ = RecordContext::NEVER;

  // Ring buffer for memory snapshot TraceEntry's
  RingBuffer<TraceEntry<StreamT>> alloc_buffer;

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<
      MempoolId_t,
      std::unique_ptr<PrivatePool<BlockT>>,
      MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool<BlockT>*, MempoolIdHash>
      graph_pools_freeable;

  // OOM callback when OOM happens.
  std::vector<OutOfMemoryObserver> oom_observers_;

  std::vector<AllocatorTraceTracker> trace_trackers_;

  // mapping from block to a stream_set, containing streams on which the block
  // was used while cudagraph capturing
  std::unordered_map<BlockT*, ska::flat_hash_set<StreamT>>
      block_to_cudagraph_stream_uses;

  // thread local compile context for each device
  static thread_local std::stack<std::string> compile_context;

 public:
 protected:
};

} // namespace c10
