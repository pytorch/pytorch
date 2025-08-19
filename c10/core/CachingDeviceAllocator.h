#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/Stream.h>
#include <c10/util/flat_hash_map.h>

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

using CaptureId_t = unsigned long long;

// first is set if the instance is created by Graph mode capture_begin.
// second is set if the instance is created by Graph mode graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

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

} // namespace c10::CachingDeviceAllocator
