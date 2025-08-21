#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Gauge.h>
#include <c10/util/flat_hash_map.h>

#include <deque>
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

inline size_t get_round_size(size_t size) {
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  }

  auto divisions = AcceleratorAllocatorConfig::roundup_power2_divisions(size);
  if (divisions > 1 && size > (kMinBlockSize * divisions)) {
    return roundup_power2_next_division(size, divisions);
  } else {
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }
}

inline size_t get_allocation_size(size_t size) {
  if (size <= kSmallSize) {
    return kSmallBuffer;
  } else if (size < kMinLargeAlloc) {
    return kLargeBuffer;
  } else {
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }
}

} // namespace c10::CachingDeviceAllocator

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMemoryCallbacksRegistry, FreeMemoryCallback);

#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMemoryCallbacksRegistry, name, __VA_ARGS__)

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

} // namespace c10

namespace c10::CachingDeviceAllocator {

// Tracing/Stats utilities

union trace_time_ {
  time_t t_;
  approx_time_t approx_t_;
};

// TraceEntry traits to map StreamT to its corresponding HandleT for different
// backends. This is mainly used to keep BC, otherwise it is better to use void*
// or c10::Stream as the generic handle.
template <typename StreamT>
struct TraceEntryTraits {
  using StreamHandleT = void*;
};

// This is a generic trace entry that can be used for any device allocator
template <typename StreamT>
struct TraceEntryBase {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via record_stream
                    // This event is generated when a free actually completes.
    SEGMENT_ALLOC, // A call to cudaMalloc to get more memory from the OS
    SEGMENT_FREE, // A call to cudaFree to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // A call to cuMemMap (used with expandable_segments)
    SEGMENT_UNMAP, // A call to unmap segment (used with expandable segments)
    SNAPSHOT, // A call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // The allocator threw an OutOfMemoryError
  };

  using StreamHandleT = typename TraceEntryTraits<StreamT>::StreamHandleT;

  TraceEntryBase(
      Action action,
      c10::DeviceIndex device,
      size_t addr,
      size_t size,
      StreamHandleT stream,
      MempoolId_t mempool,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr,
      std::string compile_context = {})
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
  StreamHandleT stream_;
  size_t size_;
  MempoolId_t mempool_;
  trace_time_ time_{};
  std::string compile_context_{};
};

using CreateContextFnPtr = std::shared_ptr<GatheredContext> (*)();

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

/**
 * Thread-safe circular buffer for storing a fixed number of entries.
 * Maintains the most recent N entries, automatically overwriting older entries
 * when the buffer reaches capacity. This is particularly useful for tracking
 * allocation traces and debugging information.
 */
template <class T>
class RingBuffer {
 public:
  RingBuffer() {
    // alloc_trace is a pointer because we need to intentionally
    // leak this on deallocation it can hold references to Python
    // state which will already be destroyed when we are in exit handlers
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    alloc_trace_ = new std::vector<T>();
  }

  // Sets the maximum number of entries the buffer can hold.
  void setMaxEntries(size_t size) {
    std::lock_guard<std::mutex> lock(alloc_trace_lock_);
    alloc_trace_max_entries_ = std::max(size_t(1), size);
  }

  // Inserts a new entry into the buffer. If buffer is full, overwrites the
  // oldest entry.
  void insertEntries(const T& entry) {
    std::lock_guard<std::mutex> lock(alloc_trace_lock_);
    if (alloc_trace_->size() < alloc_trace_max_entries_) {
      // Buffer not yet full, simply append
      alloc_trace_->emplace_back(entry);
    } else {
      // Buffer full, overwrite the oldest entry
      (*alloc_trace_)[alloc_trace_next_++] = entry;
      if (alloc_trace_next_ == alloc_trace_max_entries_) {
        alloc_trace_next_ = 0;
      }
    }
  }

  // Retrieves all entries in insertion order (oldest to newest).
  void getEntries(std::vector<T>& result) {
    std::lock_guard<std::mutex> lock(alloc_trace_lock_);
    result.reserve(alloc_trace_->size());
    result.insert(
        result.end(),
        alloc_trace_->begin() +
            static_cast<typename std::vector<T>::difference_type>(
                alloc_trace_next_),
        alloc_trace_->end());
    result.insert(
        result.end(),
        alloc_trace_->begin(),
        alloc_trace_->begin() +
            static_cast<typename std::vector<T>::difference_type>(
                alloc_trace_next_));
  }

  void clear() {
    std::lock_guard<std::mutex> lock(alloc_trace_lock_);
    alloc_trace_next_ = 0;
    alloc_trace_->clear();
  }

 private:
  // maximum capacity of the ring buffer
  size_t alloc_trace_max_entries_ = 1;

  // Both alloc_trace_ and alloc_trace_next_ needs to be used under
  // alloc_trace_lock.
  std::mutex alloc_trace_lock_;

  // Index where next entry will be written (when buffer is full)
  size_t alloc_trace_next_ = 0;

  // pointer because we need to intentionally leak this on deallocation it can
  // hold references to Python state which will already be destroyed when we
  // are in exit handlers
  std::vector<T>* alloc_trace_;
};

// Block/Pool related management utilities

template <typename BlockT>
struct BlockComparatorSize {
  bool operator()(const BlockT* a, const BlockT* b) const {
    // Note [Block Comparator]
    // Assumes all compared blocks belong to the same device (guaranteed by the
    // block pool). Without this guarantee, stream.id() could collide across
    // devices — i.e., different streams on different devices may have the same
    // stream id.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a->device == b->device);
    if (a->size != b->size) {
      return a->size < b->size;
    }
    if (a->stream.id() != b->stream.id()) {
      return a->stream.id() < b->stream.id();
    }
    return reinterpret_cast<uintptr_t>(a->ptr) <
        reinterpret_cast<uintptr_t>(b->ptr);
  }
};

template <typename BlockT>
struct BlockComparatorAddress {
  bool operator()(const BlockT* a, const BlockT* b) const {
    // see Note [Block Comparator]
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a->device == b->device);
    if (a->stream.id() != b->stream.id()) {
      return a->stream.id() < b->stream.id();
    }
    return reinterpret_cast<uintptr_t>(a->ptr) <
        reinterpret_cast<uintptr_t>(b->ptr);
  }
};

// Forward declaration
template <typename BlockT>
struct PrivatePool;

/**
 * BlockPool is a memory pool that manages reusable memory blocks of a single
 * type, such as DeviceBlock. Each instance only could contain one kind of block
 * size category (small or large).
 *
 * It is templated on BlockT, representing the type of memory block being
 * managed. The pool maintains two sets: one for currently allocated and
 * reusable blocks backed by physical memory, and another for unmapped but
 * reusable blocks managed via expandable segments.
 *
 * BlockPool also keeps track of whether it manages small or large blocks, and
 * optionally, the PrivatePool it is associated with.
 */
template <typename BlockT>
struct BlockPool {
  BlockPool(bool small, PrivatePool<BlockT>* private_pool = nullptr)
      : blocks(BlockComparatorSize<BlockT>()),
        unmapped(BlockComparatorAddress<BlockT>()),
        is_small_(small),
        owner_PrivatePool_(private_pool) {}

  // Add a Block into blocks set with updating gc counter.
  std::pair<
      typename std::set<BlockT*, BlockComparatorSize<BlockT>>::iterator,
      bool>
  insert_into_blocks(BlockT* block) {
    block->gc_count_base = get_free_blocks_call_count;
    return blocks.insert(block);
  }

  MempoolId_t owner_MempoolId() const {
    if (owner_PrivatePool_) {
      return owner_PrivatePool_->id();
    } else {
      return {0, 0};
    }
  }

  bool is_small() const {
    return is_small_;
  }

  // Returns the statistic types tracked by this BlockPool based on its size
  // category (small or large).
  StatTypes get_stat_types() const {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        is_small_ ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  // Currently allocated and reusable blocks backed by physical memory. Do not
  // insert block directly into this set. Use `insert_into_blocks` instead to
  // ensure proper handling by the garbage collection mechanism.
  std::set<BlockT*, BlockComparatorSize<BlockT>> blocks{};
  // Unmapped but reusable blocks to support expandable segments.
  std::set<BlockT*, BlockComparatorAddress<BlockT>> unmapped{};

 private:
  // Indicates whether this pool manages small or large blocks.
  bool is_small_;
  // Pointer to the PrivatePool that owns this BlockPool, if any.
  PrivatePool<BlockT>* owner_PrivatePool_;
  // Counter for the number of get_free_blocks made. This is used to track gc.
  size_t get_free_blocks_call_count{0};
};

// Forward declaration
template <typename StreamT>
struct ExpandableSegment;

/**
 * DeviceBlock is typically a fundamental unit of memory used in device caching
 * allocator. It corresponds to a memory block allocated on a specific device
 * and associated with a particular stream.
 *
 * A DeviceBlock may also track which BlockPool it belongs to. This struct is
 * intended to serve as a base type or interface that can be extended by
 * specific backend implementations.
 */
template <typename StreamT>
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

  // Constructs a DeviceBlock as a search key
  DeviceBlock(c10::DeviceIndex device, StreamT stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  // Returns the garbage collection count since this block was created.
  size_t gc_count() const {
    TORCH_INTERNAL_ASSERT(pool);
    TORCH_INTERNAL_ASSERT(pool->get_free_blocks_call_count >= gc_count_base);
    return pool->get_free_blocks_call_count - gc_count_base;
  }

  // Checks if this block is part of a split allocation.
  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  // Inserts this block between two existing blocks with [before, this, after].
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

  // Attempts to merge this block with an adjacent candidate block.And return
  // size of the candidate block in bytes, or 0 if merge failed.
  size_t try_merge(BlockT* candidate) {
    if (!candidate || candidate->allocated || candidate->event_count > 0 ||
        !candidate->stream_uses.empty() || this->mapped != candidate->mapped) {
      return 0;
    }

    TORCH_INTERNAL_ASSERT(
        this->is_split() && candidate->is_split(),
        "Both blocks must be part of split allocations");

    if (this->prev == candidate) { // [candidate this]
      // Merge with previous block: [candidate, this] -> [this]
      this->ptr = candidate->ptr;
      this->prev = candidate->prev;
      if (this->prev) {
        this->prev->next = this;
      }
      this->context_when_segment_allocated =
          std::move(candidate->context_when_segment_allocated);
    } else if (this->next == candidate) {
      // Merge with next block: [this, candidate] -> [this]
      this->next = candidate->next;
      if (this->next) {
        this->next->prev = this;
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Candidate must be adjacent block");
    }
    const size_t subsumed_size = candidate->size;
    this->size += subsumed_size;
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto& pool = *this->pool;
    auto erased = candidate->mapped ? pool.blocks.erase(candidate)
                                    : pool.unmapped.erase(candidate);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete candidate;

    return subsumed_size;
  }

  c10::DeviceIndex device;
  StreamT stream; // Allocation stream for this block
  ska::flat_hash_set<StreamT>
      stream_uses{}; // Streams that have used this block
  size_t size; // Actual block size (bytes)
  size_t requested_size; // Originally requested size (bytes)
  BlockPoolT* pool{nullptr}; // Owning memory pool
  void* ptr{nullptr}; // Memory address
  bool allocated{false}; // Whether block is currently in use
  // Virtual memory mapped to physical memory.
  // Always true unless part of expandable segment.
  // When false, block alignment matches segment size.
  bool mapped{true};
  BlockT* prev{nullptr}; // Previous block if split from a larger allocation
  BlockT* next{nullptr}; // Next block if split from a larger allocation
  int64_t event_count{0}; // Number of outstanding events referencing block
  size_t gc_count_base{0}; // Pool's gc count at block insertion time
  // Records the last time we handed this memory out from our cache
  std::shared_ptr<GatheredContext> context_when_allocated;
  // Only set for the first block in the segment (when prev == null).
  // This records the frame information when device allocation was called,
  std::shared_ptr<GatheredContext> context_when_segment_allocated;
  // Expandble segment this block belongs to
  ExpandableSegment<StreamT>* expandable_segment_{nullptr};
};

/**
 * PrivatePool manages BlockPool and their associated allocator. It maintains
 * separate BlockPools for small and large blocks, and tracks allocation usage
 * for lifecycle management.
 */
template <typename BlockT>
struct PrivatePool {
  using BlockPoolT = BlockPool<BlockT>;

  PrivatePool(MempoolId_t id, DeviceAllocator* allocator = nullptr)
      : id_(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  C10_DISABLE_COPY_AND_ASSIGN(PrivatePool);
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  DeviceAllocator* allocator() const {
    return allocator_;
  }

  MempoolId_t id() const {
    return id_;
  }

  BlockPoolT large_blocks; // Large blocks pool this PrivatePool manages
  BlockPoolT small_blocks; // Small blocks pool this PrivatePool manages
  // Number of live graphs using this pool
  int64_t use_count{1};
  // Number of unfreed device allocation made for this pool. When use_count and
  // deviceMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int64_t deviceMalloc_count{0};

 private:
  // ID of this private pool, used to identify it in the allocator.
  MempoolId_t id_;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance-critical though,
  // I'd rather not add more logic to it.
  DeviceAllocator* allocator_;
};

// Represents a contiguous virtual memory segment mapped for allocation.
struct SegmentRange {
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
  char* ptr; // Starting address of the mapped range.
  size_t size; // Size in bytes of the mapped range.
};

// ExpandableSegment traits to map StreamT to its corresponding HandleT for
// different backends. This must be specialized for each backend's stream type.
// If the backend does not support expandable segments feature, it should define
// HandleT as void* as a placeholder.
template <typename StreamT>
struct ExpandableSegmentTraits {
  using HandleT = void*;
};

/**
 * ExpandableSegment is an abstract base class that manages a virtual memory
 * segment composed of multiple equally-sized sub-segments (e.g., 2MB or 20MB)
 * on a specific device.
 *
 * It provides mechanisms for:
 * - Reserving a large virtual memory region
 * - Mapping/unmapping physical memory into sub-ranges on demand
 * - Granting access to peer devices
 * - Tracking which parts of the segment are currently allocated
 *
 * Each segment is parameterized by a Stream type (StreamT), and uses a
 * backend-specific HandleT (defined by ExpandableSegmentTraits) to represent
 * physical memory handles.
 *
 * This class is intended to be extended by backends (e.g., CUDA, XPU, etc.) to
 * implement memory mapping and access control policies. It is useful for
 * caching allocators that want to reuse virtual address ranges efficiently,
 * grow allocations dynamically, and support multi-device access through peer
 * mappings.
 */
template <typename StreamT>
struct ExpandableSegment {
  using HandleT = typename ExpandableSegmentTraits<StreamT>::HandleT;
  using PtrT = std::unique_ptr<void, std::function<void(void*)>>;

  ExpandableSegment() = default;
  C10_DISABLE_COPY_AND_ASSIGN(ExpandableSegment);
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment& operator=(ExpandableSegment&&) = delete;

  // Initializes the segment with the specified device, stream, segment size...
  // This function must be called only once per instance.
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
    max_handles_ = numSegments(getReservedVirtualMemorySize(device));
    TORCH_INTERNAL_ASSERT(
        !ptr_, "ExpandableSegment::init() has already been called");
    void* ptr = nullptr;
    createVirtualMemoryAddress(&ptr);
    ptr_ = PtrT(ptr, [this](void* p) {
      if (p)
        this->releaseVirtualMemoryAddress(p);
    });
  }

  // Maps a virtual memory range to physical memory.
  virtual SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
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
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  // Returns the base pointer of the virtual memory segment.
  char* ptr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<char*>(ptr_.get());
  }

  // Returns the total size of the virtual memory segment.
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
  }

 private:
  // Runtime-related methods that must be implemented by derived classes.

  // Returns the reserved virtual memory size for this segment, which may be
  // larger than the total size if the segment is expandable.
  virtual size_t getReservedVirtualMemorySize(c10::DeviceIndex device) = 0;

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

  // Internal methods for segment calculations and iteration

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
  PtrT ptr_{nullptr};
  // Size of each segment in bytes.
  size_t segment_size_{0};
  // Maximum number of segments that can be allocated in this segment.
  size_t max_handles_{0};
  // Physical memory handles for the segments.
  std::vector<std::optional<HandleT>> handles_{};
  // Peer devices on which this memory should be mapped and accessible.
  std::vector<c10::DeviceIndex> peers_{};
};

// Factory function to create and initialize an ExpandableSegment instance.
template <
    typename ExpandableSegmentT,
    typename = std::enable_if_t<std::is_base_of_v<
        ExpandableSegment<typename ExpandableSegmentT::StreamT>,
        ExpandableSegmentT>>>
ExpandableSegmentT* make_expandable_segment(
    c10::DeviceIndex device,
    std::optional<typename ExpandableSegmentT::StreamT> stream,
    size_t segment_size,
    std::vector<c10::DeviceIndex> peers) {
  ExpandableSegmentT* ptr = new ExpandableSegmentT();
  TORCH_INTERNAL_ASSERT(ptr, "Failed to allocate memory for ExpandableSegment");
  ptr->init(device, std::move(stream), segment_size, std::move(peers));
  return ptr;
}

template <typename StreamT, typename BlockT = DeviceBlock<StreamT>>
struct AllocParams {
  using BlockPoolT = BlockPool<BlockT>;

  enum Status : uint8_t { Ok, OOM, Error };

  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      StreamT stream,
      BlockPoolT* pool,
      size_t alloc_size)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        stat_types(pool->get_stat_types()) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }

  StreamT stream() const {
    return search_key.stream;
  }

  size_t size() const {
    return search_key.size;
  }

  BlockPoolT* pool; // Pointer to the BlockPool managing this allocation
  size_t alloc_size; // Size of the allocation in bytes
  StatTypes stat_types; // Types of statistics to track for this allocation
  BlockT* block{nullptr}; // Pointer to the allocated block, if found/created
  Status status{Status::Ok}; // Status of the allocation attempt

 private:
  BlockT search_key; // Key used to search for an existing block in the pool
};

template <
    typename StreamT,
    typename EventT,
    typename BlockT = DeviceBlock<StreamT>,
    typename ExpandableSegmentT = ExpandableSegment<StreamT>,
    typename = std::enable_if_t<
        std::is_base_of_v<ExpandableSegment<StreamT>, ExpandableSegmentT>>>
struct CachingDeviceAllocatorImpl {
  using BlockPoolT = BlockPool<BlockT>;
  using PrivatePoolT = PrivatePool<BlockT>;
  using AllocParamsT = AllocParams<StreamT, BlockT>;
  using TraceEntryT = TraceEntryBase<StreamT>;

  virtual ~CachingDeviceAllocatorImpl() = default;

  CachingDeviceAllocatorImpl(c10::DeviceIndex device_index)
      : device_index_(device_index),
        large_blocks(/*small=*/false),
        small_blocks(/*small=*/true) {
    stats.max_split_size =
        static_cast<int64_t>(AcceleratorAllocatorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

 private:
  /* Internal methods for processing runtime */

  // Deallocate a device memory pointer associated with the given block.
  virtual void deallocate_device_ptr(BlockT* block) = 0;

  // Record an event on stream and return it. Note this function may be called
  // under allocator lock.
  virtual EventT record_event_for_stream(StreamT stream) = 0;

  // Queries the status of an event. Returns true if the event is complete.
  // Note this function may be called under allocator lock.
  virtual bool query_event(const EventT& event) = 0;

  // Synchronizes the given event, blocking until it completes.
  virtual void synchronize_event(const EventT& event) = 0;

  // Records events for all streams that have used the given memory block.
  // This function transfers the ownership of the block’s `stream_uses` set.
  void insert_events(BlockT* block) {
    c10::DeviceGuard device_guard(block->device);

    ska::flat_hash_set<StreamT> streams(std::move(block->stream_uses));
    TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
    block->event_count += streams.size();
    for (auto& stream : streams) {
      EventT event = record_event_for_stream(stream);
      outstanding_events[stream].emplace_back(std::move(event), block);
    }
  }

  // Removes stream uses that were added to the block during graph capture.
  // And restores the block's `stream_uses` set to its state before the
  // capture.
  void remove_graph_stream_uses(BlockT* block) {
    // remove stream uses added during cudagraph capture
    // (i.e., block->stream_uses - block->cudagraph_stream_uses)
    if (C10_UNLIKELY(
            block_to_graph_stream_uses.find(block) !=
            block_to_graph_stream_uses.end())) {
      ska::flat_hash_set<StreamT> streams(std::move(block->stream_uses));
      TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
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
        // only streams recorded before cudagraph will be used to insert
        // events since we know all streams recorded during cudagraph must
        // have completed (refer to Section 3.2.8.7.3.1 Cross-stream
        // Dependencies and Events in CUDA Programming Guide).
        remove_graph_stream_uses(block);
        insert_events(block);
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  // Processes all outstanding events and frees associated memory blocks when
  // safe.
  void process_events(const std::shared_ptr<GatheredContext>& context) {
    insert_events_deferred_until_no_capture(context);

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = outstanding_events.begin();
         it != outstanding_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventT event = std::move(e.first);
        BlockT* block = e.second;

        if (!query_event(event)) {
          // Return the ownership of the Event (unique ptr)
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
        it = outstanding_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Synchronizes all outstanding events across all streams and frees associated
  // memory blocks when safe.
  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context,
      PrivatePoolT* pool = nullptr) {
    // Synchronize on outstanding events and then free associated blocks.
    stats.num_sync_all_streams++;

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway.empty());
    insert_events_deferred_until_no_capture(context);

    for (auto it = outstanding_events.begin();
         it != outstanding_events.end();) {
      for (auto e = it->second.begin(); e != it->second.end();) {
        BlockT* block = e->second;

        // If a pool was passed, only synchronize the events
        // that are associated with the pool, otherwise move on
        if (pool && block->pool->owner_PrivatePool != pool) {
          ++e;
          continue;
        }

        EventT event = std::move(e->first);

        synchronize_event(event);

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        // We are done with the event, so erase it from the deque
        e = it->second.erase(e);
      }

      // If the events deque is empty, only then erase it from the map
      if (it->second.empty()) {
        it = outstanding_events.erase(it);
      } else {
        it++;
      }
    }
  }

  /* Internal methods for managing device blocks */

  // Moves a block into a pool of cached free blocks
  void free_block(
      BlockT* block,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    record_trace(
        TraceEntryT::FREE_COMPLETED,
        block->ptr,
        block->requested_size,
        block->stream,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

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
    // Makes sure the Block* isn't already present in the pool we're freeing
    // it back into. NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = block->pool->insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = block->pool->get_stat_types();

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

  // Releases a block back to the system.
  void release_block(
      BlockT* block,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    stats.num_device_free++;
    record_trace(
        TraceEntryT::SEGMENT_FREE,
        block->ptr,
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
      deallocate_device_ptr(block);
    }
    total_allocated_memory -= block->size;

    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->deviceMalloc_count > 0);
      pool->owner_PrivatePool->deviceMalloc_count--;
    }

    StatTypes stat_types = pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.segment[stat_type].decrease(1);
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

    auto reserved_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.reserved_bytes);
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    if (block->size >= AcceleratorAllocatorConfig::max_split_size())
      stats.oversize_segments.decrease(1);

    auto earsed = pool->blocks.erase(block);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(earsed == 1);
    delete block;
  }

  // Release one or more oversize blocks to the system allocator. But only
  // enough to satisfy the target size.
  bool release_available_cached_blocks(
      const AllocParamsT& p,
      const std::shared_ptr<GatheredContext>& context) {
    if (AcceleratorAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;

    BlockPoolT& pool = *p.pool;
    BlockT key(p.device(), p.stream(), p.size());
    if (key.size < AcceleratorAllocatorConfig::max_split_size())
      key.size = AcceleratorAllocatorConfig::max_split_size();

    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream() ||
        (*it)->expandable_segment_) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;

      size_t totalReleased = 0;
      --it; // Back up one item. Now on the largest block for the correct stream

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

  // Attempts to find a free block in the pool that matches the search key.
  bool get_free_block(AllocParamsT& p) {
    BlockPoolT& pool = *p.pool;

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
      if (AcceleratorAllocatorConfig::use_expandable_segments() &&
          is_expandable_segment_supported()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](BlockT* b) {
          // b->next may belong to pool.unmapped (reserved but not mapped)
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

  // Returns the BlockPool (large or small, default or private) for the given
  // size and stream.
  BlockPoolT& get_pool(size_t size, StreamT stream) {
    // captures_underway is a conservative guess that the current stream may
    // be capturing. It's only non-empty if some thread has begun and not yet
    // ended a capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        if (entry.second(stream)) {
          auto it1 = graph_pools.find(entry.first);
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          return (size <= kSmallSize) ? it1->second->small_blocks
                                      : it1->second->large_blocks;
        }
      }
    }
    // Graph mode isn't underway, so we can use the default pools.
    return (size <= kSmallSize) ? small_blocks : large_blocks;
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
    size_t total_age = 0;
    int64_t freeable_block_count = 0;
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
      double age_threshold = static_cast<double>(total_age) /
          static_cast<double>(freeable_block_count);
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

  /* Internal methods for managing expandable segments */

  virtual bool is_expandable_segment_supported() const {
    return false;
  }

  /* Internal methods for utils: tracing, stats, gc... */

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  virtual void record_trace(
      typename TraceEntryT::Action action,
      size_t addr,
      size_t size,
      c10::Stream stream,
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::shared_ptr<GatheredContext> context) {
    if (!record_history && trace_trackers_.empty())
      return;
    std::string compile_string = "N/A";
    if (!compile_context.empty()) {
      compile_string = compile_context.top();
    }
    auto te = TraceEntryT(
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

  bool trigger_free_memory_callbacks(AllocParamsT& p) {
    bool freed_memory = false;
    for (const auto& name : FreeMemoryCallbacksRegistry()->Keys()) {
      freed_memory |= FreeMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  /* Internal members */

  c10::DeviceIndex device_index_;

  // lock around all operations
  mutable std::recursive_mutex mutex;

  // unallocated cached blocks larger than 1 MB
  BlockPoolT large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPoolT small_blocks;

  // Allocated or in use by a stream. Holds all active allocations, whether
  // they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<BlockT*> active_blocks;

  // Outstanding events that are waiting for the used streams to complete.
  ska::flat_hash_map<StreamT, std::deque<std::pair<EventT, BlockT*>>>
      outstanding_events;

  bool set_fraction = false; // Whether to set the fraction feature.

  size_t total_allocated_memory = 0; // Records total allocated memory

  size_t allowed_memory_maximum = 0; // Equals to (total memory * fraction)

  DeviceStats stats; // device statistics

  bool record_history = false;

  std::atomic<CreateContextFnPtr> context_recorder_{nullptr};

  RecordContext record_context_ = RecordContext::NEVER;

  // Trace buffer for recording AllocatorTraceTracker for call back.
  std::vector<std::function<void(const TraceEntryT&)>> trace_trackers_;

  // Thread local compile context for each device
  static thread_local std::stack<std::string> compile_context;

  // Ring buffer for memory snapshot TraceEntry's
  RingBuffer<TraceEntryT> alloc_buffer;

  // Tracks if we are diverting some allocations to a specific pool. Most of
  // the time it's empty, in which case malloc can avoid calling query
  // streams' capture state API such as `cudaStreamGetCaptureInfo` in the hot
  // path.
  std::vector<std::pair<MempoolId_t, std::function<bool(StreamT)>>>
      captures_underway;

  // Members specific to Graph mode capture.

  // Private pools for Graph feature
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePoolT>, MempoolIdHash>
      graph_pools;

  // Holds free blocks whose event insertion is deferred until capture
  // finished.
  std::vector<BlockT*> needs_events_deferred_until_no_capture;

  // Mapping from block to a stream set, containing streams on which the block
  // was used while graph feature capturing
  std::unordered_map<BlockT*, ska::flat_hash_set<StreamT>>
      block_to_graph_stream_uses;
};

} // namespace c10::CachingDeviceAllocator
