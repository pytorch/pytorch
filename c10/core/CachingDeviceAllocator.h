#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Gauge.h>
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

  StatTypes get_stat_types() {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
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

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge(BlockT* candidate) {
    if (!candidate || candidate->allocated || candidate->event_count > 0 ||
        !candidate->stream_uses.empty() || this->mapped != candidate->mapped) {
      return 0;
    }

    AT_ASSERT(this->is_split() && candidate->is_split());

    if (this->prev == candidate) { // [candidate this]
      this->ptr = candidate->ptr;
      this->prev = candidate->prev;
      if (this->prev) {
        this->prev->next = this;
      }
      this->context_when_segment_allocated =
          std::move(candidate->context_when_segment_allocated);
    } else { // [this candidate]
      this->next = candidate->next;
      if (this->next) {
        this->next->prev = this;
      }
    }
    const size_t subsumed_size = candidate->size;
    this->size += subsumed_size;
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto& pool = *this->pool;
    auto erased =
    candidate->mapped ? pool.blocks.erase(candidate) : pool.unmapped.erase(candidate);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete candidate;

    return subsumed_size;
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

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class C10_CUDA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

template <
    typename StreamT,
    typename EventT,
    typename HandleT = void*,
    typename BlockT = DeviceBlock<StreamT, HandleT>,
    typename ES = ExpandableSegment<StreamT, HandleT>,
    typename c10::DeviceType device_type>
struct CachingDeviceAllocatorImpl {
  virtual ~CachingDeviceAllocatorImpl() = default;

  static constexpr c10::DeviceType static_device_type = device_type;

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

    AllocParams<StreamT, HandleT> params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = pool.get_stat_types();

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              AcceleratorAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks(context);
      }
      // Attempt allocate
      // WARNING: alloc_block may release the allocator lock when calling
      // cudaMalloc. So far this function has not modified allocator state, but
      // keep in mind that any observed allocator state may change across calls
      // to alloc_block since it may release the lock.
      block_found = alloc_block(params, false, context, lock)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params, context) &&
              alloc_block(params, false, context, lock))
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway.empty()) &&
              release_cached_blocks(context, {0, 0}) &&
              alloc_block(params, true, context, lock));
    }

    // we are about to oom, try to use existing mempools as a last resort
    if (!block_found && params.err == AllocParams<StreamT, HandleT>::Status::OOM) {
      // if already trying to use a mempool, then just oom
      bool active_pool = params.pool->owner_PrivatePool;
      if (!active_pool) {
        for (MempoolId_t mempool_id : use_on_oom_pools) {
          auto tid = std::this_thread::get_id();
          auto filter = [tid](cudaStream_t) {
            return std::this_thread::get_id() == tid;
          };
          beginAllocateToPool(mempool_id, filter);
          auto& mempool = get_pool(size, stream);
          AllocParams<StreamT, HandleT> mempool_params(
              device, size, stream, &mempool, alloc_size);
          mempool_params.stat_types = mempool.get_stat_types();
          block_found = get_free_block(mempool_params);
          endAllocateToPool(mempool_id);
          releasePool(mempool_id);
          if (block_found) {
            params = mempool_params;
            break;
          }
        }
      }
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.status == AllocParams<StreamT, HandleT>::Status::OOM);

      size_t device_free = 0;
      size_t device_total = 0;
      device_free, device_total = getMemoryInfo();
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      std::string proc_info = reportProcessMemoryInfo(device);

      record_trace(
          TraceEntry::OOM,
          device_free,
          params.size(),
          params.stream(),
          params.device(),
          params.pool->owner_MempoolId(),
          std::move(context));
      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
          static_cast<int64_t>(size),
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(static_device_type, device));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

      size_t allocated_in_private_pools = 0;
      auto get_size_block = [](const BlockPool<BlockT>& pool) {
        size_t res = 0;
        for (const auto& block : pool.blocks) {
          res += block->size;
        }
        return res;
      };
      for (const auto& p : graph_pools) {
        allocated_in_private_pools += get_size_block(p.second->large_blocks);
        allocated_in_private_pools += get_size_block(p.second->small_blocks);
      }

      std::string private_pool_msg;

      if (allocated_in_private_pools > 0) {
        private_pool_msg = "with " + format_size(allocated_in_private_pools) +
            " allocated in private pools (e.g., CUDA Graphs), ";
      }

      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();

      for (const auto& obs : observers_local) {
        obs(device,
            alloc_size,
            set_fraction ? allowed_memory_maximum : device_total,
            device_free);
      }

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "Out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". GPU ",
          static_cast<int>(device),
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. ",
          proc_info,
          allowed_info,
          "Of the allocated memory ",
          format_size(allocated_bytes + allocated_in_private_pools),
          " is allocated by PyTorch, ",
          private_pool_msg,
          "and ",
          format_size(
              reserved_bytes - allocated_bytes - allocated_in_private_pools),
          " is reserved by PyTorch but unallocated.",
          " If reserved but unallocated memory is large try setting",
          " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid"
          " fragmentation.  See documentation for Memory Management "
          " (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        params, orig_size, std::move(context), split_remainder);
  }

   /** Returns a copy of the memory allocator stats **/
   DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void free(BlockT* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = *block->pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocation[stat_type].decrease(1);
      stats.allocated_bytes[stat_type].decrease(block->size);
    });
#define CREATE_GAUGE(key)                              \
    []() -> ::c10::monitor::GaugeHandle& {             \
      static ::c10::monitor::GaugeHandle handle(key);  \
      return handle;                                   \
    }()
    std::string_view key = "pytorch." +
      c10::DeviceTypeName(static_device_type) +
      "CachingAllocator.allocated_bytes";
    auto allocated_bytes_gauge = CREATE_GAUGE(key);
    allocated_bytes_gauge.record(
        stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    record_trace(
        TraceEntry::FREE_REQUESTED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_allocated);

    if (block->size >= AcceleratorAllocatorConfig::max_split_size())
      stats.oversize_allocations.decrease(1);

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(!captures_underway.empty())) {
        // It's forbidden to cudaEventQuery an event recorded during CUDA graph
        // capture. We conservatively defer recording end-of-life events until
        // the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
      free_block(block, context);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -static_cast<int64_t>(orig_block_size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(static_device_type, block->device));
  }

  void recordStream(BlockT* block, StreamT stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
    if (C10_UNLIKELY(!captures_underway.empty())) {
      block_to_cudagraph_stream_uses[block].insert(stream);
    }
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache(MempoolId_t mempool_id) {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context, mempool_id);
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocation[statType].reset_accumulated();
      stats.segment[statType].reset_accumulated();
      stats.active[statType].reset_accumulated();
      stats.inactive_split[statType].reset_accumulated();
      stats.allocated_bytes[statType].reset_accumulated();
      stats.reserved_bytes[statType].reset_accumulated();
      stats.active_bytes[statType].reset_accumulated();
      stats.inactive_split_bytes[statType].reset_accumulated();
      stats.requested_bytes[statType].reset_accumulated();
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    stats.num_sync_all_streams = 0;
    stats.num_device_alloc = 0;
    stats.num_device_free = 0;
    stats.oversize_allocations.reset_accumulated();
    stats.oversize_segments.reset_accumulated();
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocation[statType].reset_peak();
      stats.segment[statType].reset_peak();
      stats.active[statType].reset_peak();
      stats.inactive_split[statType].reset_peak();
      stats.allocated_bytes[statType].reset_peak();
      stats.reserved_bytes[statType].reset_peak();
      stats.active_bytes[statType].reset_peak();
      stats.inactive_split_bytes[statType].reset_peak();
      stats.requested_bytes[statType].reset_peak();
    }
    stats.oversize_allocations.reset_peak();
    stats.oversize_segments.reset_peak();
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

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(
      const AllocParams& p,
      const std::shared_ptr<GatheredContext>& context) {
    if (AcceleratorAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool<BlockT>& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    // Use constructor for search key.
    BlockT key(p.search_key.device, p.search_key.stream, p.search_key.size);
    key.size = (key.size < AcceleratorAllocatorConfig::max_split_size())
        ? AcceleratorAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream() ||
        (*it)->expandable_segment_) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
           ((*it)->size >= AcceleratorAllocatorConfig::max_split_size()) &&
           ((*it)->stream == p.stream())) {
        auto cur = it;
        bool is_first = cur == pool.blocks.begin();
        if (!is_first) {
          --it;
        }
        if (!(*cur)->expandable_segment_) {
          release_block(*cur, context);
          totalReleased += (*cur)->size;
        }
        if (is_first) {
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
   } else {
      release_block(*it, context);
    }
    return true;
  }

  bool release_cached_blocks(
      const std::shared_ptr<GatheredContext>& context,
      MempoolId_t mempool_id) {
    if (mempool_id.first == 0 && mempool_id.second == 0 &&
        captures_underway.empty()) {
      // If there is no active mempool, we work on releasing *all* blocks.

      // First ensure that all blocks that can't currently be allocated due to
      // outstanding events are returned to the pool.
      synchronize_and_free_events(context);

      // Free all non-split cached blocks to system allocator
      release_blocks(large_blocks, context);
      release_blocks(small_blocks, context);
    }

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      if (mempool_id.first != 0 || mempool_id.second != 0) {
        if (it->first == mempool_id) {
          // If there is an active mempool, we sync only the events
          // associated with the pool
          synchronize_and_free_events(context, it->second);
        } else {
          // otherwise we move on
          ++it;
          continue;
        }
      }
      // See notifyCaptureDestroy for the strategy here.
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks, context);
      release_blocks(it->second->large_blocks, context);
      if (it->second->cudaMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  bool get_free_block(AllocParams<BlockT>& p) {
    BlockPool<BlockT>& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            AcceleratorAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      ++pool.get_free_blocks_call_count;
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;

    if ((*it)->expandable_segment_) {
      if (AcceleratorAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](BlockT* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Rarely expandable segments has been turned off after we have
        // already allocated some blocks as expandable. For instance,
        // since we cannot share expandable memory via IPC, someone might
        // temporarily disable it. In this case we need to honor this request
        // by only finding non-expandable blocks
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // Do not return an oversized block for a large request
    if ((p.size() < AcceleratorAllocatorConfig::max_split_size()) &&
        ((*it)->size >= AcceleratorAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= AcceleratorAllocatorConfig::max_split_size()) &&
        ((*it)->size >=
         p.size() + AcceleratorAllocatorConfig::max_non_split_rounding_size()))
      return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  // This function assumes that global lock has been taken while calling into
  // this function. We do cudaMalloc sync call in this function which
  // can be expensive while holding the lock. Hence, we pass-in the lock to the
  // function to temporarily release the lock before cudaMalloc call and acquire
  // it back again after the call so that other threads dont get blocked.
  bool alloc_block(
      AllocParams<StreamT>& p,
      bool isRetry,
      const std::shared_ptr<GatheredContext>& ctx,
      std::unique_lock<std::recursive_mutex>& lock) {

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
#ifdef FBCODE_CAFFE2
    bool in_fbcode = true;
#else
    bool in_fbcode = false;
#endif

    bool active_pool =
        p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator();
    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.status = AllocParams<StreamT>::OOM;
      return false;
      
    } 
    
    // Temporarily disable checkpointing & cudagraphs internally
    if (
      is_expandable_segment_enabled() &&
        !(in_fbcode && p.pool->owner_PrivatePool)) {
      TORCH_CHECK(
          !active_pool,
          "torch.cuda.MemPool doesn't currently support expandable_segments.");
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.status = AllocParams<StreamT>::Ok;
        if (p.pool->owner_PrivatePool) {
          // The block is for a CUDA graph's PrivatePool.
          p.pool->owner_PrivatePool->deviceMalloc_count++;
        }
      } else {
        p.status = AllocParams<StreamT>::OOM;
      }
      return bool(p.block);
    }

    if (p.status == AllocParams<StreamT>::OOM) {
      if(allocate_block_ptr(p, &ptr, size, lock) == AllocParams<StreamT>::OOM) {
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->deviceMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new BlockT(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      stats.segment[stat_type].increase(1);
      stats.reserved_bytes[stat_type].increase(size);
    });
    if (size >= AllocatorAllocatorConfig::max_split_size())
      stats.oversize_segments.increase(1);

#define CREATE_GAUGE(key)                            \
      []() -> ::c10::monitor::GaugeHandle& {             \
        static ::c10::monitor::GaugeHandle handle(key);  \
        return handle;                                   \
      }()
    
    std::string_view key = "pytorch." +
      c10::DeviceTypeName(static_device_type) +
      "CachingAllocator.reserved_bytes";
    auto reserved_bytes_gauge = CREATE_GAUGE(key);
    
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    stats.num_device_alloc++;
    record_trace(
        TraceEntry::SEGMENT_ALLOC,
        int64_t(p.block->ptr),
        p.block->size,
        p.stream(),
        p.device(),
        p.pool->owner_MempoolId(),
        ctx);
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  BlockT* alloc_found_block(
      const AllocParams<StreamT>& params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.status == AllocParams<StreamT>::Ok && params.block != nullptr &&
        params.block->ptr != nullptr);
    BlockT* block = params.block;
    BlockT* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new BlockT(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
      // [new block, original block]
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      bool inserted = pool->insert_into_blocks(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split && !block->expandable_segment_) {
        // An already-split inactive block is being shrunk by size bytes.
        decrease_stat_array(
            stats.inactive_split_bytes, block->size, params.stat_types);
      } else if (!block->expandable_segment_) {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          stats.inactive_split_bytes[stat_type].increase(remaining->size);
          stats.inactive_split[stat_type].increase(1);
        });
      }

    } else if (already_split && !block->expandable_segment_) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        stats.inactive_split_bytes[stat_type].decrease(block->size);
        stats.inactive_split[stat_type].decrease(1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;

    block->context_when_allocated = std::move(context);
    record_trace(
        TraceEntry::ALLOC,
        int64_t(block->ptr),
        orig_size,
        block->stream,
        block->device,
        block->pool->owner_MempoolId(),
        block->context_when_allocated);

    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      stats.allocation[stat_type].increase(1);
      stats.allocated_bytes[stat_type].increase(block->size);
      stats.active[stat_type].increase(1);
      stats.active_bytes[stat_type].increase(block->size);
      stats.requested_bytes[stat_type].increase(block->requested_size);
    });
    if (block->size >= AcceleratorAllocatorConfig::max_split_size())
      stats.oversize_allocations.increase(1);

#define CREATE_GAUGE(key)                            \
  []() -> ::c10::monitor::GaugeHandle& {             \
    static ::c10::monitor::GaugeHandle handle(key);  \
    return handle;                                   \
  }()

    std::string_view key = "pytorch." +
        c10::DeviceTypeName(static_device_type) +
        "CachingAllocator.allocated_bytes";
    auto allocated_bytes_gauge = CREATE_GAUGE(key);
    allocated_bytes_gauge.record(
        stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(static_device_type, device));

    return block;
  }
  
  void garbage_collect_cached_blocks(
    const std::shared_ptr<GatheredContext>& context) {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
      AcceleratorAllocatorConfig::garbage_collection_threshold() *
      static_cast<double>(allowed_memory_maximum));
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    size_t total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count();
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
         freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold =
        static_cast<double>(total_age) / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        BlockT* block = *it;
        ++it;
        if (!block->is_split() && !block->expandable_segment_ &&
            static_cast<double>(block->gc_count()) >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count(); // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block, context);
        }
      }
    }
  }

  void release_block(
    BlockT* block,
    const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    stats.num_device_free++;
    record_trace(
      TraceEntry::SEGMENT_FREE,
      int64_t(block->ptr),
      block->size,
      block->stream,
      block->device,
      block->pool->owner_MempoolId(),
      context ? context : block->context_when_segment_allocated);

    auto* pool = block->pool;
    if (pool->owner_PrivatePool && pool->owner_PrivatePool->allocator()) {
      // If there is an active mempool with a given allocator,
      // we use the given allocator's delete function.
      pool->owner_PrivatePool->allocator()->raw_delete((void*)block->ptr);
    } else {
      release_block_ptr(block);
    }
    total_allocated_memory -= block->size;

    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.segment[stat_type].decrease(1);
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

#define CREATE_GAUGE(key)                            \
  []() -> ::c10::monitor::GaugeHandle& {             \
    static ::c10::monitor::GaugeHandle handle(key);  \
    return handle;                                   \
  }()

    std::string_view key = "pytorch." +
        c10::DeviceTypeName(static_device_type) +
        "CachingAllocator.reserved_bytes";
    auto reserved_bytes_gauge = CREATE_GAUGE(key);
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    if (block->size >= AcceleratorAllocatorConfig::max_split_size())
      stats.oversize_segments.decrease(1);
    pool->blocks.erase(block);
    delete block;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(
    BlockT* block,
    const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    record_trace(
        TraceEntry::FREE_COMPLETED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<BlockT*, 2> merge_candidates = {block->prev, block->next};
    for (BlockT* merge_candidate : merge_candidates) {
      const auto subsumed_size = block->try_merge(merge_candidate);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= static_cast<int64_t>(subsumed_size);
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = pool.get_stat_types();

    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segments from
      // inactive_split
      if (!block->expandable_segment_) {
        if (net_change_inactive_split_blocks > 0) {
          stats.inactive_split[stat_type].increase(
              static_cast<size_t>(net_change_inactive_split_blocks));
        } else if (net_change_inactive_split_blocks < 0) {
          stats.inactive_split[stat_type].decrease(
              static_cast<size_t>(-net_change_inactive_split_blocks));
        }
        if (net_change_inactive_split_size > 0) {
          stats.inactive_split_bytes[stat_type].increase(
              static_cast<size_t>(net_change_inactive_split_size));
        } else if (net_change_inactive_split_size < 0) {
          stats.inactive_split_bytes[stat_type].decrease(
              static_cast<size_t>(-net_change_inactive_split_size));
        }
      }
      stats.active[stat_type].decrease(1);
      stats.active_bytes[stat_type].decrease(original_block_size);
      stats.requested_bytes[stat_type].decrease(requested_size);
    });
  }

  void remove_graph_stream_uses(BlockT* block) {
    // remove stream uses added during cudagraph capture
    // (i.e., block->stream_uses - block->cudagraph_stream_uses)
    if (C10_UNLIKELY(
      block_to_graph_stream_uses.find(block) !=
      block_to_graph_stream_uses.end())) {
      ska::flat_hash_set<StreamT> streams(std::move(block->stream_uses));
      AT_ASSERT(block->stream_uses.empty());
      for (auto& stream : streams) {
        if (block_to_graph_stream_uses[block].find(stream) ==
        block_to_graph_stream_uses[block].end()) {
          block->stream_uses.insert(stream);
        }
      }
      block_to_graph_stream_uses.erase(block);
    }
  }

  void insert_events_deferred_until_no_capture(
    const std::shared_ptr<GatheredContext>& context) {
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        // only streams recorded before cudagraph will be used to insert events
        // since we know all streams recorded during cudagraph must have
        // completed (refer to Section 3.2.8.7.3.1 Cross-stream Dependencies and
        // Events in CUDA Programming Guide).
        remove_graph_stream_uses(block);
        insert_events(block);
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events(const std::shared_ptr<GatheredContext>& context) {
    insert_events_deferred_until_no_capture(context);

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;
        
        if(!query_event(event)) {
          e.first = std::move(event);
          break;
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Called by CUDAGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (auto it = captures_underway.begin(); it != captures_underway.end();
         ++it) {
      if (it->first == mempool_id) {
        captures_underway.erase(it);
        return;
      }
    }
    TORCH_CHECK(
        false, "endAllocatePool: not currently recording to mempool_id");
  }

  PrivatePool<BlockT>* get_private_pool(MempoolId_t mempool_id) {
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    return it->second.get();
  }

  // Called by CUDAGraph::reset and MemPool::~MemPool()
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated cudaGraphExec_t has been destroyed. We can't blindly
    // delete and cudaFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_blocks it may now
    // cudaFree blocks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto pp = get_private_pool(mempool_id);
    auto uc = --(pp->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      bool inserted = graph_pools_freeable.insert({mempool_id, pp}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context,
      PrivatePool<BlockT>* pool = nullptr) {
    // Synchronize on outstanding events and then free associated blocks.
    stats.num_sync_all_streams++;

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway.empty());
    insert_events_deferred_until_no_capture(context);

    for (auto it = block_events.begin(); it != block_events.end();) {
      for (auto e = it->second.begin(); e != it->second.end();) {
        BlockT* block = e->second;

        // If a pool was passed, only synchronize the events
        // that are associated with the pool, otherwise move on
        if (pool && block->pool->owner_PrivatePool != pool) {
          ++e;
          continue;
        }

        EventT event = std::move(e->first);

        synchronize_event(*event);

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        // We are done with the event, so erase it from the deque
        e = it->second.erase(e);
      }

      // If the events deque is empty, only then erase the
      // cuda event from the events map
      if (it->second.empty()) {
        it = block_events.erase(it);
      } else {
        it++;
      }
    }
  }

  void record_trace(
    TraceEntry::Action action,
    size_t addr,
    size_t size,
    StreamT stream,
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::shared_ptr<GatheredContext> context) {
    if (!record_history && trace_trackers_.empty())
      return;
    std::string compile_string = "N/A";
    if (!compile_context.empty()) {
      compile_string = compile_context.top();
    }
    auto te = TraceEntry<StreamT>(
      action,
      device,
      addr,
      size,
      stream,
      mempool_id,
      getApproximateTime(),
      record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr,
      compile_string);

    // Callbacks should not include any Pytorch call
    for (const auto& cb : trace_trackers_) {
      cb(te);
    }

    if (record_history) {
      alloc_buffer.insertEntries(te);
    }
  }

  bool should_split(const BlockT* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small || is_expandable_segment_enabled()) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < AcceleratorAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  virtual bool is_expandable_segment_enabled() const {
    return false;
  }

  virtual void insert_events(BlockT* block) = 0;

  // This function is used to trigger free memory callbacks
  virtual bool trigger_free_memory_callbacks(AllocParams<StreamT>& p) {
    return false;
  }

  /* The runtime-related APIs */
  virtual void release_block_ptr(BlockT* block) = 0;

  virtual std::pair<size_t, size_t> getMemoryInfo() = 0;

  virtual bool query_event(EventT& event) = 0;

  virtual void synchronize_event(EventT& event) = 0;

  std::string reportProcessMemoryInfo(c10::DeviceIndex device) = 0;

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

  bool set_fraction = false;

  // all live expandable segments
  std::vector<ExpandableSegment<StreamT, HandleT>*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

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
      block_to_graph_stream_uses;

  // thread local compile context for each device
  static thread_local std::stack<std::string> compile_context;

 public:
 protected:
};

} // namespace c10
