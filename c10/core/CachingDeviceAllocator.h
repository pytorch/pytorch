#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Gauge.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>

#include <deque>
#include <set>
#include <stack>
#include <thread>

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

  // Allocates memory of size nbytes on the device.
  virtual void* raw_alloc(size_t nbytes) = 0;

  // Deallocates memory previously allocated by raw_alloc.
  virtual void raw_delete(void* ptr) = 0;

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

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated; // per-watcher context
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct GenericSegmentInfo {
  c10::DeviceIndex device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0; // unrounded, actually requested size
  size_t allocated_size = 0;
  size_t active_size = 0;
  c10::Stream stream;
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0};
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
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

  CachingDeviceAllocatorImpl(
      c10::DeviceIndex device_index,
      c10::DeviceType device_type)
      : device_index_(device_index),
        device_type_(device_type),
        large_blocks(/*small=*/false),
        small_blocks(/*small=*/true) {
    stats.max_split_size =
        static_cast<int64_t>(AcceleratorAllocatorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

  virtual BlockT* malloc(DeviceIndex device, size_t orig_size, StreamT stream) {
    // done outside the lock because we don't know what locks the recorder
    // needs to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.empty())) {
      // Reclaim allocations used on multiple streams if their GPU-side uses
      // are complete.
      //
      // NOTE: We skip process_events if a capture might be underway because
      // process_events performs event queries which are illegal during graph
      // capture.
      //
      // Dumb simple solution: defer reclaiming these allocations until after
      // capture. Cross-stream memory use is uncommon, so the deferral's effect
      // on memory use during capture should be small.
      process_events(context);
    }

    const size_t size = get_round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParamsT params(device, size, stream, &pool, alloc_size);

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
              AcceleratorAllocatorConfig::garbage_collection_threshold() >
                  0.0)) {
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
    if (!block_found && params.status == AllocParamsT::OOM) {
      // if already trying to use a mempool, then just oom
      bool active_pool = params.pool->owner_PrivatePool;
      if (!active_pool) {
        for (MempoolId_t mempool_id : use_on_oom_pools) {
          auto tid = std::this_thread::get_id();
          auto filter = [tid](StreamT) {
            return std::this_thread::get_id() == tid;
          };
          beginAllocateToPool(mempool_id, filter);
          auto& mempool = get_pool(size, stream);
          AllocParamsT mempool_params(
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
      TORCH_INTERNAL_ASSERT(params.status == AllocParamsT::OOM);

      size_t device_free = 0;
      size_t device_total = 0;
      getMemoryInfo(device, device_free, device_total);
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      std::string proc_info = reportProcessMemoryInfo(device);

      record_trace(
          TraceEntryT::OOM,
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
          c10::Device(device_type_, device));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

      size_t allocated_in_private_pools = 0;
      auto get_size_block = [](const BlockPoolT& pool) {
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
          " PYTORCH_ALLOC_CONF=expandable_segments:True to avoid"
          " fragmentation.  See documentation for Memory Management "
          " (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        params, orig_size, std::move(context), split_remainder);
  }

  virtual void free(BlockT* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = block->pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocation[stat_type].decrease(1);
      stats.allocated_bytes[stat_type].decrease(block->size);
    });
    auto allocated_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.allocated_bytes);
    allocated_bytes_gauge.record(
        stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    record_trace(
        TraceEntryT::FREE_REQUESTED,
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
        c10::Device(device_type_, block->device));
  }

  virtual void recordStream(BlockT* block, StreamT stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
    if (C10_UNLIKELY(!captures_underway.empty())) {
      block_to_graph_stream_uses[block].insert(stream);
    }
  }

  /** returns cached blocks to the system allocator **/
  virtual void emptyCache(MempoolId_t mempool_id) {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context, mempool_id);
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  virtual void resetAccumulatedStats() {
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
  virtual void resetPeakStats() {
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

  // Returns the size of free and total memory in bytes on the device.
  virtual void getMemoryInfo(
      c10::DeviceIndex device,
      size_t& free_bytes,
      size_t& total_bytes) = 0;

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  void attachAllocatorTraceTracker(
      std::function<void(const TraceEntryT&)> tracker) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    trace_trackers_.emplace_back(std::move(tracker));
  }

  // Get memory fraction limiting maximum allocated memory
  virtual double getMemoryFraction() {
    if (!set_fraction) {
      return 1.0;
    }

    size_t device_free = 0;
    size_t device_total = 0;
    getMemoryInfo(device_index_, device_free, device_total);
    return static_cast<double>(allowed_memory_maximum) /
        static_cast<double>(device_total);
  }

  // Set memory fraction to limit maximum allocated memory
  virtual void setMemoryFraction(double fraction) {
    size_t device_free = 0;
    size_t device_total = 0;
    getMemoryInfo(device_index_, device_free, device_total);
    allowed_memory_maximum =
        static_cast<size_t>(fraction * static_cast<double>(device_total));
    set_fraction = true;
  }

  void setUseOnOOM(MempoolId_t mempool_id) {
    // Choose if this pool should be used as a last resort before ooming
    std::lock_guard<std::recursive_mutex> lock(mutex);
    use_on_oom_pools.insert(mempool_id);
  }

  void createOrIncrefPool(MempoolId_t mempool_id, DeviceAllocator* allocator) {
    // Create a PrivatePool object if it does not exist yet
    // and increment its use_count
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id, allocator);
  }

  int getPoolUseCount(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto pp = get_private_pool(mempool_id);
    return pp->use_count;
  }

  // See Note [Interaction with CUDA graph capture]

  // Called by Graph::capture_begin
  virtual void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(StreamT)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id);
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by Graph::capture_end
  virtual void endAllocateToPool(MempoolId_t mempool_id) {
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

  // Called by Graph::reset and MemPool::~MemPool()
  virtual void releasePool(MempoolId_t mempool_id) {
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

  std::vector<c10::DeviceIndex> peers() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return devices_with_peer_access_;
  }

  void addPeerAccess(c10::DeviceIndex dev_to_access) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (std::find(
            devices_with_peer_access_.begin(),
            devices_with_peer_access_.end(),
            dev_to_access) != devices_with_peer_access_.end()) {
      return;
    }
    devices_with_peer_access_.push_back(dev_to_access);
    for (auto& es : expandable_segments_) {
      es->addPeer(dev_to_access);
    }
  }

 private:
  /* Internal methods for processing runtime */

  // Deallocate a device memory pointer associated with the given block.
  virtual void deallocate_device_ptr(BlockT* block) = 0;

  // Allocate a device memory pointer of the given size and parameters.
  virtual void allocate_device_ptr(void** ptr, AllocParamsT& p) = 0;

  virtual std::string reportProcessMemoryInfo(c10::DeviceIndex device) {
    return "";
  }

  // Allocate a device memory pointer for a primitive type.
  virtual void allocPrimitive(void** ptr, AllocParamsT& p) {
    if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
      *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(p.alloc_size);
      p.status = *ptr ? AllocParamsT::Ok : AllocParamsT::OOM;
    } else {
      // Handle p.status inside allocate_device_ptr
      allocate_device_ptr(ptr, p);
    }
  }

  // Allocate a device memory pointer with an optional lock.
  virtual void mallocMaybeCapturingWithOptionalLock(
      void** ptr,
      AllocParamsT& p,
      std::unique_lock<std::recursive_mutex>& lock) = 0;

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

  // This function assumes that global lock has been taken while calling into
  // this function. We do cudaMalloc sync call in this function which
  // can be expensive while holding the lock. Hence, we pass-in the lock to the
  // function to temporarily release the lock before cudaMalloc call and acquire
  // it back again after the call so that other threads dont get blocked.
  bool alloc_block(
      AllocParamsT& p,
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
      p.status = AllocParamsT::OOM;
      return false;
      // Temporarily disable checkpointing & cudagraphs internally
    } else if (
        AcceleratorAllocatorConfig::use_expandable_segments() &&
        is_expandable_segment_supported() &&
        !(in_fbcode && p.pool->owner_PrivatePool)) {
      TORCH_CHECK(
          !active_pool,
          "MemPool doesn't currently support expandable_segments.");
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.status = AllocParamsT::Ok;
        if (p.pool->owner_PrivatePool) {
          // The block is for a CUDA graph's PrivatePool.
          p.pool->owner_PrivatePool->deviceMalloc_count++;
        }
      } else {
        p.status = AllocParamsT::OOM;
      }
      return bool(p.block);
    } else {
      mallocMaybeCapturingWithOptionalLock(&ptr, p, lock);
      if (p.status == AllocParamsT::OOM) {
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

    if (size >= AcceleratorAllocatorConfig::max_split_size())
      stats.oversize_segments.increase(1);

    auto reserved_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.reserved_bytes);
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    stats.num_device_alloc++;
    record_trace(
        TraceEntryT::SEGMENT_ALLOC,
        p.block->ptr,
        p.block->size,
        p.stream(),
        p.device(),
        p.pool->owner_MempoolId(),
        ctx);
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  BlockT* alloc_found_block(
      const AllocParamsT& params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.status == AllocParamsT::OOM && params.block != nullptr &&
        params.block->ptr != nullptr);
    BlockT* block = params.block;
    BlockT* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new BlockT(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
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
        TraceEntryT::ALLOC,
        block->ptr,
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

    auto allocated_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.allocated_bytes);
    allocated_bytes_gauge.record(
        stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(device_type_, device));

    return block;
  }

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

  void release_blocks(
      BlockPoolT& pool,
      const std::shared_ptr<GatheredContext>& context) {
    std::vector<BlockT*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      BlockT* block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block, context);
      }
    }
    for (BlockT* block : to_unmap) {
      unmap_block(block, context);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
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
      // If mempool_id == (0,0), release all cached blocks from both the default
      // and private pools (global cleanup). Otherwise, release only the cached
      // blocks in the private pool associated with the given mempool_id.
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

      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks, context);
      release_blocks(it->second->large_blocks, context);
      if (it->second->deviceMalloc_count == 0) {
        auto erased = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erased == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
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

  std::vector<BlockT*> get_all_blocks() const {
    std::vector<BlockT*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  bool should_split(const BlockT* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        (AcceleratorAllocatorConfig::use_expandable_segments() &&
         is_expandable_segment_supported())) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < AcceleratorAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  /* Internal methods for managing expandable segments */

  virtual bool is_expandable_segment_supported() const {
    return false;
  }

  void release_expandable_segment(BlockT* block) {
    TORCH_INTERNAL_ASSERT(
        block->size == block->expandable_segment_->size(),
        "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);
    auto it = std::find(
        expandable_segments_.begin(),
        expandable_segments_.end(),
        block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    delete block;
  }

  // Returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  BlockT* find_expandable_block(
      c10::DeviceIndex device,
      StreamT stream,
      BlockPoolT* pool,
      size_t size) {
    BlockT key(device, stream, 0);

    auto allocatable = [](BlockT* b) {
      return b && !b->allocated && b->event_count == 0 &&
          b->stream_uses.empty();
    };
    auto has_available_address_space = [&](BlockT* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };

    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream;
         ++it) {
      BlockT* c = *it;
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        // The preceding segment block may be a free and mapped block in
        // BlockPool.blocks and its size could be smaller than requested,
        // but it may be expandable.
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        // The block c's size may be smaller than requested size, but it is
        // expandable.
        return c;
      }
    }

    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(
        make_expandable_segment<ExpandableSegmentT>(
            device, stream, segment_size, devices_with_peer_access_));

    ExpandableSegmentT* es = expandable_segments_.back();
    BlockT* candidate = new BlockT(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  BlockT* try_allocate_expandable_block(
      c10::DeviceIndex device,
      StreamT stream,
      BlockPoolT* pool,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    BlockT* candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(
              new_candidate, std::min(remaining, candidate->next->size), ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }

  // Map the block, split off any unused portion if necessary, then attempt to
  // merge the resulting block with its adjacent previous and next blocks.
  bool map_block(
      BlockT* to_map,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated); // unmapped blocks should not keep
                                          // history
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(
        mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

    BlockPoolT& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      BlockT* remaining = new BlockT(
          to_map->device,
          to_map->stream,
          to_map->size - mapped_range.size,
          &pool,
          static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    to_map->try_merge(to_map->prev);
    to_map->try_merge(to_map->next);
    pool.insert_into_blocks(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = to_map->pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(mapped_range.size);
    });

    auto reserved_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.reserved_bytes);
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    stats.num_device_alloc++;
    record_trace(
        TraceEntryT::SEGMENT_MAP,
        mapped_range.ptr,
        mapped_range.size,
        to_map->stream,
        to_map->device,
        to_map->pool->owner_MempoolId(),
        ctx);
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = ctx;
    }

    return true;
  }

  void unmap_block(
      BlockT* block,
      const std::shared_ptr<GatheredContext>& context) {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size = unmapped.ptr - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      BlockT* before_free = new BlockT(
          block->device, block->stream, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->insert_into_blocks(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      BlockT* after_free = new BlockT(
          block->device,
          block->stream,
          after_size,
          block->pool,
          static_cast<char*>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->insert_into_blocks(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    block->try_merge(block, block->prev);
    block->try_merge(block, block->next);
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = block->pool->get_stat_types();
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(unmapped.size);
    });

    auto reserved_bytes_gauge =
        STATIC_GAUGE(pytorch.CachingAllocator.reserved_bytes);
    reserved_bytes_gauge.record(
        stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
            .current);

    if (block->pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(
          block->pool->owner_PrivatePool->deviceMalloc_count > 0);
      block->pool->owner_PrivatePool->deviceMalloc_count--;
    }

    stats.num_device_free++;
    record_trace(
        TraceEntryT::SEGMENT_UNMAP,
        unmapped.ptr,
        unmapped.size,
        block->stream,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_segment_allocated);
  }

  /* Internal methods for private pool for Graph feature */

  void create_or_incref_pool(
      MempoolId_t mempool_id,
      DeviceAllocator* allocator = nullptr) {
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool.
      // Make a new pool for CUDAGraph capture or torch.cuda.use_mem_pool
      // usage. use_count is initially 1, which means the pool is
      // being used since somebody called createOrIncrefPool.
      graph_pools.emplace(
          mempool_id, std::make_unique<PrivatePoolT>(mempool_id, allocator));
    } else {
      // mempool_id references an existing pool, which the current CUDAGraph
      // capture or torch.cuda.use_mem_pool will
      // share. Check this pool is live (at least one other capture already
      // references it). Increment it to establish the usage.
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      TORCH_INTERNAL_ASSERT(allocator == nullptr);
      it->second->use_count++;
    }
  }

  PrivatePoolT* get_private_pool(MempoolId_t mempool_id) {
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    return it->second.get();
  }

  std::vector<BlockT*> get_private_pool_head_blocks(PrivatePoolT* pool) const {
    std::vector<BlockT*> blocks;
    for (BlockT* b : active_blocks) {
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    for (BlockT* b : pool->small_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }
    for (BlockT* b : pool->large_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    return blocks;
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

  void pushCompileContext(std::string& md) {
    compile_context.push(md);
  }

  void popCompileContext() {
    if (!compile_context.empty()) {
      compile_context.pop();
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

  c10::DeviceType device_type_;

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

  // Tracks which pools we can use as a last resort before ooming
  ska::flat_hash_set<MempoolId_t, MempoolIdHash> use_on_oom_pools;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

  // Tracks if we are diverting some allocations to a specific pool. Most of
  // the time it's empty, in which case malloc can avoid calling query
  // streams' capture state API such as `cudaStreamGetCaptureInfo` in the hot
  // path.
  std::vector<std::pair<MempoolId_t, std::function<bool(StreamT)>>>
      captures_underway;

  // All live expandable segments
  std::vector<ExpandableSegmentT*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  // Members specific to Graph mode capture.

  // Private pools for Graph feature
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePoolT>, MempoolIdHash>
      graph_pools;

  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePoolT*, MempoolIdHash>
      graph_pools_freeable;

  // Holds free blocks whose event insertion is deferred until capture
  // finished.
  std::vector<BlockT*> needs_events_deferred_until_no_capture;

  // Mapping from block to a stream set, containing streams on which the block
  // was used while graph feature capturing
  std::unordered_map<BlockT*, ska::flat_hash_set<StreamT>>
      block_to_graph_stream_uses;
};

// Calls made by record_function will save annotations
struct AnnotationEntry {
  AnnotationEntry(c10::DeviceIndex device, approx_time_t time)
      : device_(device) {
    time_.approx_t_ = time;
  }

  void recordUserMetadata(const std::string& name, std::string value) {
    metadata_[name] = std::move(value);
  }

  c10::DeviceIndex device_;
  trace_time_ time_{};
  std::unordered_map<std::string, std::string> metadata_;
};

template <
    c10::DeviceType deviceType,
    c10::DeleterFnPtr deleteFunc,
    typename ImplT,
    typename BaseDeviceAllocator = c10::DeviceAllocator>
struct CachingDeviceAllocatorInterface : BaseDeviceAllocator {
  using BlockT = typename ImplT::BlockT;
  using StreamT = typename ImplT::StreamT;
  using ExpandableSegmentT = typename ImplT::ExpandableSegmentT;

  void malloc(
      void** devPtr,
      c10::DeviceIndex device,
      size_t size,
      StreamT stream) {
    checkDeviceIndex(device);
    BlockT* block = nullptr;
    block = impls_[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          deviceType, reinterpret_cast<uintptr_t>(*devPtr));
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
          deviceType, reinterpret_cast<uintptr_t>(block->ptr));
    }
    impls_[block->device]->free(block);
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }

    c10::impl::VirtualGuardImpl impl(deviceType);
    c10::Device device = impl.getDevice();
    void* devPtr = nullptr;
    c10::Stream stream = impl.getStream(device);
    malloc(&devPtr, device, nbytes, StreamT(stream));

    return devPtr;
  }

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  DeleterFnPtr raw_deleter() const override {
    return deleteFunc;
  }

  void recordStream(const DataPtr& ptr, StreamT stream) {
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
    checkDeviceIndex(device);
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
  void init(c10::DeviceIndex device_count) override {
    const auto size = static_cast<c10::DeviceIndex>(impls_.size());
    if (size < device_count) {
      impls_.resize(device_count);
      for (const auto& i : c10::irange(size, device_count)) {
        impls_[i] = std::make_unique<ImplT>(i, deviceType);
      }
    }
  }

  bool initialized() override {
    return !impls_.empty();
  }

  double getMemoryFraction(c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    return impls_[device]->getMemoryFraction();
  }

  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    TORCH_CHECK_VALUE(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within [0, 1].");
    impls_[device]->setMemoryFraction(fraction);
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    for (auto& impl : impls_) {
      impl->attachOutOfMemoryObserver(observer);
    }
  }

  void attachAllocatorTraceTracker(
      std::function<void(const typename ImplT::TraceEntryT&)> tracker)
      override {
    for (auto& impl : impls_) {
      impl->attachAllocatorTraceTracker(tracker);
    }
  }

  void createOrIncrefPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      DeviceAllocator* allocator) override {
    checkDeviceIndex(device);
    impls_[device]->createOrIncrefPool(std::move(mempool_id), allocator);
  }

  void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    checkDeviceIndex(device);
    impls_[device]->setUseOnOOM(std::move(mempool_id));
  }

  // Graph interactions
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(StreamT)> filter) override {
    checkDeviceIndex(device);
    impls_[device]->beginAllocateToPool(
        std::move(mempool_id), std::move(filter));
  }

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    checkDeviceIndex(device);
    impls_[device]->endAllocateToPool(mempool_id);
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    checkDeviceIndex(device);
    impls_[device]->releasePool(std::move(mempool_id));
  }

  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    checkDeviceIndex(device);
    return impls_[device]->getPoolUseCount(std::move(mempool_id));
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

  static size_t get_mutex_shard_id(void* ptr) {
    return twang_mix64(reinterpret_cast<size_t>(ptr)) % kNumMutexShard;
  }

  // A prime number close to 64, used for sharding to reduce contention.
  static constexpr size_t kNumMutexShard = 67;

  // Aligns the mutex to avoid false sharing on cache lines.
  struct alignas(hardware_destructive_interference_size) AlignedMutex {
    std::mutex m;
  };
  std::array<AlignedMutex, kNumMutexShard> mutex;

  // A map of allocated blocks, sharded by mutex to reduce contention.
  std::array<ska::flat_hash_map<void*, BlockT*>, kNumMutexShard>
      allocated_blocks{};

  // Per-device allocator implementations.
  std::vector<std::unique_ptr<ImplT>> impls_{};
};

} // namespace c10::CachingDeviceAllocator
