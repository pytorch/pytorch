#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <deque>
#include <mutex>
#include <set>
#include <vector>

namespace c10::xpu::XPUCachingAllocator {

using namespace c10::CachingAllocator;
using namespace c10::CachingDeviceAllocator;

// newly allocated memory with 512-byte alignment.
constexpr size_t kDeviceAlignment = 512;

namespace {
using stream_set = ska::flat_hash_set<xpu::XPUStream>;

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
bool BlockComparatorSize(const Block* a, const Block* b);
bool BlockComparatorAddress(const Block* a, const Block* b);

struct PrivatePool;

struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
  PrivatePool* owner_PrivatePool;

  MempoolId_t owner_MempoolId() const;
};

struct ExpandableSegment;

struct Block {
  DeviceIndex device;
  sycl::queue* queue{nullptr}; // underlying queue of the allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // True if this Block is backed by physical pages
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding XPU events
  // The stack context at the time this block was most recently allocated from
  // the caching pool
  std::shared_ptr<GatheredContext> context_when_allocated;
  // Only set for the first block in a segment (when prev == nullptr). Record
  // the stack context at the time the underlying memory segment was first
  // allocated (e.g., via sycl::aligned_alloc_device and map).
  std::shared_ptr<GatheredContext> context_when_segment_allocated;
  ExpandableSegment* expandable_segment{nullptr}; // owning expandable segment

  Block(
      DeviceIndex device,
      sycl::queue* queue,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        queue(queue),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(DeviceIndex device, sycl::queue* queue, size_t size)
      : device(device), queue(queue), size(size), requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  // Inserts this block between two existing blocks with [before, this, after].
  void splice(Block* before, Block* after) {
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
};

bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->queue != b->queue) {
    return reinterpret_cast<uintptr_t>(a->queue) <
        reinterpret_cast<uintptr_t>(b->queue);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->queue != b->queue) {
    return reinterpret_cast<uintptr_t>(a->queue) <
        reinterpret_cast<uintptr_t>(b->queue);
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

// Represents a contiguous virtual memory segment mapped for allocation.
struct SegmentRange {
  SegmentRange(void* addr, size_t bytes)
      : ptr(static_cast<char*>(addr)), size(bytes) {}
  char* ptr; // Starting address of the mapped range.
  size_t size; // Size in bytes of the mapped range.
};

struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<sycl::queue*> queue,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers)
      : device_(device),
        queue_(queue),
        // 2MB for small pool, 20MB for large pool
        segment_size_(segment_size),
        peers_(std::move(peers)) {
    const auto device_total =
        c10::xpu::get_raw_device(device)
            .get_info<sycl::info::device::global_mem_size>();
    // The extra 1/8 allows flexibility for remapping or moving pages within the
    // segment when unmapping earlier regions.
    constexpr float kVirtualMemOversubscriptFactor = 1.125f; // 1 + 1/8
    max_handles_ = numSegments(static_cast<size_t>(
        static_cast<float>(device_total) * kVirtualMemOversubscriptFactor));
    ptr_ = sycl::ext::oneapi::experimental::reserve_virtual_mem(
        segment_size_ * max_handles_, xpu::get_device_context());
  }

  C10_DISABLE_COPY_AND_ASSIGN(ExpandableSegment);
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment& operator=(ExpandableSegment&&) = delete;

  // Maps a virtual memory range to physical memory.
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }

    // Ensure handles_ vector is large enough to hold all segments.
    if (end > handles_.size()) {
      handles_.resize(end, std::nullopt);
    }

    // Allocate and map physical memory for each segment.
    for (const auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      auto& handle = handles_.at(i);
      try {
        // Allocate physical memory for each segment. Construct the physical_mem
        // in-place to avoid copies.
        auto& mem = handle.emplace(
            xpu::get_raw_device(device_),
            xpu::get_device_context(),
            segment_size_);
        // Map the allocated physical memory into the virtual address space.
        mem.map(
            ptr_ + i * segment_size_,
            segment_size_,
            sycl::ext::oneapi::experimental::address_access_mode::read_write);
      } catch (const sycl::exception& e) {
        // Allocation failure: typically sycl::errc::memory_allocation.
        // Mapping failure: typically sycl::errc::runtime (e.g., OOM due to
        // over-subscription).
        // Note: constructing physical_mem may over-subscribe device memory but
        // not immediately trigger OOM. The actual OOM can occur during map().
        // Roll back all segments allocated or mapped in this operation.
        handle.reset();
        for (const auto j : c10::irange(begin, i)) {
          sycl::ext::oneapi::experimental::unmap(
              // NOLINTNEXTLINE(performance-no-int-to-ptr)
              reinterpret_cast<void*>(ptr_ + segment_size_ * j),
              segment_size_,
              xpu::get_device_context());
          handles_.at(j).reset();
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
    }
    return rangeFromHandles(begin, end);
  }

  // Unmap a virtual memory range from physical memory.
  SegmentRange unmap(SegmentRange range) {
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
    return reinterpret_cast<char*>(ptr_);
  }

  // Returns the total size of the virtual memory segment.
  size_t size() const {
    return max_handles_ * segment_size_;
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    sycl::ext::oneapi::experimental::free_virtual_mem(
        ptr_, segment_size_ * max_handles_, xpu::get_device_context());
  }

 private:
  // Unmaps the physical memory handles in the range [begin, end) from the
  // segment.
  void unmapHandles(size_t begin, size_t end) {
    // Currently, we don't support IPC shared memory with expandable segments.
    TORCH_INTERNAL_ASSERT(queue_);
    // As explained in Note [Safe to Free Blocks on BlockPool], additional
    // synchronization is unnecessary here because the memory is already safe to
    // release.
    for (const auto i : c10::irange(begin, end)) {
      // Note: physical_mem's destructor does NOT automatically unmap any mapped
      // ranges. Users must explicitly call unmap on all ranges before
      // destroying the physical_mem object.
      sycl::ext::oneapi::experimental::unmap(
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<void*>(ptr_ + segment_size_ * i),
          segment_size_,
          xpu::get_device_context());
      // Here physical_mem object is being destructed.
      handles_.at(i) = std::nullopt;
    }
    trimHandles();
  }

  // Remove trailing unused handles from the end of handles_.
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  // Iterates over all contiguous ranges of allocated segments in `handles_`,
  // and invokes the provided function `fn(start, end)` for each range.
  // Each range is defined as a half-open interval [start, end).
  void forEachAllocatedRange(const std::function<void(size_t, size_t)>& fn) {
    size_t start = 0;
    for (const auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  // Returns the number of full segments required to cover `size` bytes.
  // Rounds up to ensure partial segments are counted.
  size_t numSegments(size_t size) const {
    return (size + segment_size_ - 1) / segment_size_;
  }

  // Returns the index of the segment that contains the pointer `p`,
  // relative to the base pointer `ptr_`. This is the *inclusive* lower bound
  // of the segment that includes `p`.
  size_t segmentLeft(char* p) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(p >= ptr() && p < ptr() + size());
    size_t offset = p - ptr();
    return offset / segment_size_;
  }

  // Returns the index of the segment just *past* the one containing pointer
  // `p`, relative to the base pointer `ptr_`. This is the *exclusive* upper
  // bound, useful for [begin, end) style ranges.
  // If `p` lies exactly on a segment boundary, this is equal to segmentLeft(p).
  // Otherwise, it rounds up and returns segmentLeft(p) + 1.
  size_t segmentRight(char* p) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(p >= ptr() && p < ptr() + size());
    size_t offset = p - ptr();
    return numSegments(offset);
  }

  // Constructs a SegmentRange spanning indices [start, end).
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  c10::DeviceIndex device_{-1};
  std::optional<sycl::queue*> queue_;
  // Virtual memory address used for reservation.
  uintptr_t ptr_{0};
  // Size of each segment in bytes.
  size_t segment_size_{0};
  // Maximum number of segments that can be allocated in this segment.
  size_t max_handles_{0};
  // Physical memory handles for the segments.
  std::vector<std::optional<sycl::ext::oneapi::experimental::physical_mem>>
      handles_;
  // Peer devices on which this memory could be accessible, reserved.
  std::vector<c10::DeviceIndex> peers_;
};

struct AllocParams {
  AllocParams(
      DeviceIndex device,
      size_t size,
      sycl::queue* queue,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, queue, size), pool(pool), alloc_size(alloc_size) {}

  DeviceIndex device() const {
    return search_key.device;
  }

  sycl::queue* queue() const {
    return search_key.queue;
  }

  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block{nullptr};
  StatTypes stat_types = {};
};

// Internal implementation that manages actual memory blocks.
// high level MemPool interface wraps PrivatePool via MempoolId.
struct PrivatePool {
  PrivatePool(MempoolId_t id, XPUAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  // default Mempool when no Mempool is specified
  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed allocations made for this pool. When use_count and
  // allocation_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int allocation_count{0};
  XPUAllocator* allocator_;
  BlockPool large_blocks;
  BlockPool small_blocks;

 public:
  XPUAllocator* allocator() {
    return allocator_;
  }
};

MempoolId_t BlockPool::owner_MempoolId() const {
  if (owner_PrivatePool) {
    return owner_PrivatePool->id;
  } else {
    return {0, 0};
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

void allocPrimitive(void** ptr, size_t size, AllocParams& p) {
  if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
    *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(size);
  } else {
    *ptr = sycl::aligned_alloc_device(
        kDeviceAlignment,
        size,
        xpu::get_raw_device(p.device()),
        xpu::get_device_context());
  }
}

void deletePrimitive(void* ptr, BlockPool* pool) {
  if (pool->owner_PrivatePool && pool->owner_PrivatePool->allocator()) {
    pool->owner_PrivatePool->allocator()->raw_delete(ptr);
  } else {
    sycl::free(ptr, xpu::get_device_context());
  }
}

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
    alloc_trace_max_entries_ = std::max(static_cast<size_t>(1), size);
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

  void getEntries(std::vector<T>& result) const {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    result.reserve(result.size() + alloc_trace->size());
    std::rotate_copy(
        alloc_trace->begin(),
        std::next(alloc_trace->begin(), alloc_trace_next),
        alloc_trace->end(),
        std::back_inserter(result));
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
  mutable std::mutex alloc_trace_lock;
  size_t alloc_trace_next = 0;
  // Leaked on deallocation to avoid issues with Python shutdown
  std::vector<T>* alloc_trace;
};

} // anonymous namespace

class DeviceCachingAllocator {
 private:
  mutable std::recursive_mutex mutex;
  DeviceStats stats;
  BlockPool large_blocks; // unallocated cached blocks larger than 1 MB
  BlockPool small_blocks; // unallocated cached blocks 1 MB or smaller
  ska::flat_hash_set<Block*> active_blocks; // allocated or in use by a stream
  ska::flat_hash_map<xpu::XPUStream, std::deque<std::pair<sycl::event, Block*>>>
      xpu_events;
  DeviceIndex device_index;
  size_t allowed_memory_maximum = 0;
  bool set_fraction = false;
  std::vector<ExpandableSegment*> expandable_segments;
  std::vector<c10::DeviceIndex> devices_with_peer_access; // reserved
  bool record_history = false;
  std::atomic<CreateContextFn> context_recorder_;
  RecordContext record_context_ = RecordContext::NEVER;
  RingBuffer<TraceEntry> alloc_buffer;
  std::unordered_set<TraceEntry::Action> skip_actions_list;
  std::vector<std::pair<MempoolId_t, std::function<bool(sycl::queue*)>>>
      captures_underway;
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    TORCH_INTERNAL_ASSERT(dst->is_split() && src->is_split());
    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else { // [dst src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  std::vector<Block*> get_all_blocks() const {
    std::vector<Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  void free_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    record_trace(
        TraceEntry::FREE_COMPLETED,
        reinterpret_cast<size_t>(block->ptr),
        block->requested_size,
        block->queue,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_allocated);
    block->context_when_allocated = nullptr;

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    auto& pool = *block->pool;
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const auto subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= static_cast<int64_t>(subsumed_size);
      }
    }

    active_blocks.erase(block);
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // `inactive_split` is intended to reflect blocks that cannot be
      // immediately released even when a free is requested. However, expandable
      // segments behave differently: any fully free page within an expandable
      // segment can always be safely released. Accurately tracking expandable
      // segments in this statistic would significantly complicate the
      // bookkeeping logic, so we intentionally exclude expandable segments from
      // the `inactive_split` metric.
      if (!block->expandable_segment) {
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

  void process_events(const std::shared_ptr<GatheredContext>& context) {
    using namespace sycl::info;
    for (auto it = xpu_events.begin(); it != xpu_events.end();) {
      while (!it->second.empty()) {
        auto& e = it->second.front();
        auto event = e.first;
        auto* block = e.second;
        if (event.get_info<event::command_execution_status>() !=
            event_command_status::complete) {
          break;
        }
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = xpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return AcceleratorAllocatorConfig::large_segment_size();
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  BlockPool& get_pool(size_t size, sycl::queue* queue) {
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        // lookup for mempool id matching current capture graph
        if (entry.second(queue)) {
          auto it1 = graph_pools.find(entry.first);
          // lookup mempool
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

  // Finds the first (lowest-address) block in any segment that has sufficient
  // contiguous free virtual address space to satisfy `size`. The available
  // space may span multiple adjacent blocks, which can include both free and
  // unmapped segments.
  Block* find_expandable_block(
      c10::DeviceIndex device,
      sycl::queue* queue,
      BlockPool* pool,
      size_t size) {
    Block key(device, queue, 0);

    auto allocatable = [](Block* b) {
      return b && !b->allocated && b->event_count == 0 &&
          b->stream_uses.empty();
    };
    auto has_available_address_space = [&](Block* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->queue == queue;
         ++it) {
      Block* c = *it;
      // The unmapped block might have a free mapped block right before it.
      // By starting from the previous block, we can use both:
      // [Free Mapped Block] + [Unmapped Block] = More contiguous space
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
    auto segment_size = pool->is_small
        ? kSmallBuffer
        : AcceleratorAllocatorConfig::large_segment_size();
    expandable_segments.emplace_back(new ExpandableSegment(
        device, queue, segment_size, devices_with_peer_access));

    ExpandableSegment* es = expandable_segments.back();
    Block* candidate = new Block(device, queue, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(
      Block* to_map,
      size_t size,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated,
        "Unapped blocks must no hold allocation context.");
    auto mapped_range =
        to_map->expandable_segment->map(SegmentRange{to_map->ptr, size});
    // Failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(
        mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block* remaining = new Block(
          to_map->device,
          to_map->queue,
          to_map->size - mapped_range.size,
          &pool,
          static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment = to_map->expandable_segment;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.blocks.insert(to_map);

    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(mapped_range.size);
    });

    record_trace(
        TraceEntry::SEGMENT_MAP,
        reinterpret_cast<size_t>(mapped_range.ptr),
        mapped_range.size,
        to_map->queue,
        to_map->device,
        to_map->pool->owner_MempoolId(),
        context);
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = context;
    }

    return true;
  }

  Block* try_allocate_expandable_block(
      c10::DeviceIndex device,
      sycl::queue* queue,
      BlockPool* pool,
      size_t size,
      const std::shared_ptr<GatheredContext>& context) {
    // Candidate points to the start of a chain of contiguous blocks with
    // sufficient virtual address space (>= size). The chain may consist of:
    // Case 1: [Unmapped Block] -> null
    // Case 2: [Unmapped Block] -> [Free Mapped Block]
    // Case 3: [Free Mapped Block] -> [Unmapped Block]
    Block* candidate = find_expandable_block(device, queue, pool, size);

    // Map first block if unmapped (Case 1 & 2), use std::min to avoid
    // over-mapping.
    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), context)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    // Map additional blocks until we have enough continuous space (Case 3).
    // Each map_block() call merges newly mapped blocks with adjacent free
    // blocks
    while (candidate->size < size) {
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      // Map only what we need from the `new_candidate` block.
      if (!map_block(
              new_candidate,
              std::min(remaining, new_candidate->size),
              context)) {
        return nullptr;
      }
      candidate = new_candidate;
    }

    // Remove from the free pool; block will be marked as `allocated` in
    // alloc_found_block()
    pool->blocks.erase(candidate);
    return candidate;
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
      return false;
    }
    if ((*it)->expandable_segment) {
      if (AcceleratorAllocatorConfig::use_expandable_segments()) {
        // When expandable segments are enabled, consider both the current block
        // and any immediately adjacent unmapped region as a single expandable
        // area. For "best fit" allocation, we use the total expandable size
        // instead of just the block's current size, so that blocks which can
        // grow into a larger contiguous range are preferred.
        auto expandable_size = [](Block* b) {
          // b->next may belong to pool.unmapped (reserved but not mapped)
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        // Looks for the best fit block with expandable size.
        while ((*it)->expandable_segment && next != pool.blocks.end() &&
               (*next)->queue == p.queue() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Expandable segments were previously enabled, but are now disabled
        // (e.g. to avoid IPC issues). Skip any expandable blocks and only
        // find from regular non-expandable segments.
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment &&
                 (*it)->queue == p.queue());
        if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
          return false;
        }
      }
    }
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<GatheredContext>& context) {
    auto size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
    bool active_pool =
        p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator();
    if (set_fraction &&
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current +
                size >
            allowed_memory_maximum) {
      return false;
    } else if (AcceleratorAllocatorConfig::use_expandable_segments()) {
      TORCH_CHECK(
          !active_pool,
          "torch.xpu.MemPool doesn't currently support expandable_segments.");
      p.block = try_allocate_expandable_block(
          p.device(), p.queue(), p.pool, p.size(), context);
      if (p.block && p.pool->owner_PrivatePool) {
        // The block is used only for XPU graph's PrivatePool.
        p.pool->owner_PrivatePool->allocation_count++;
      }
      return bool(p.block);
    } else {
      allocPrimitive(&ptr, size, p);
      if (!ptr) {
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      p.pool->owner_PrivatePool->allocation_count++;
    }
    p.block = new Block(p.device(), p.queue(), size, p.pool, ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      stats.segment[stat_type].increase(1);
      stats.reserved_bytes[stat_type].increase(size);
    });
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);

    record_trace(
        TraceEntry::SEGMENT_ALLOC,
        reinterpret_cast<size_t>(p.block->ptr),
        p.block->size,
        p.queue(),
        p.device(),
        p.pool->owner_MempoolId(),
        context);
    p.block->context_when_segment_allocated = context;

    return true;
  }

  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context,
      PrivatePool* pool = nullptr) {
    for (auto& xe : xpu_events) {
      for (auto& e : xe.second) {
        auto event = e.first;
        auto* block = e.second;
        if (pool && block->pool->owner_PrivatePool != pool) {
          continue;
        }
        event.wait();
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
    }
    xpu_events.clear();
  }

  void release_expandable_segment(Block* block) {
    // See Note [Safe to Free Blocks on BlockPool], additional synchronization
    // is unnecessary here because this function is only called by
    // release_cached_blocks().
    TORCH_INTERNAL_ASSERT(
        block->size == block->expandable_segment->size(),
        "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);

    auto it = std::find(
        expandable_segments.begin(),
        expandable_segments.end(),
        block->expandable_segment);
    TORCH_INTERNAL_ASSERT(it != expandable_segments.end());

    expandable_segments.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment;
    delete block;
  }

  void release_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * Callers must ensure that all accesses to the block, whose raw pointer is
     * allocated by SYCL APIs, have been completed before invoking sycl::free.
     *
     * We have to do a device-level synchronization before free these blocks to
     * guarantee that all kernels can access to the blocks have finished.
     */
    TORCH_INTERNAL_ASSERT(!block->expandable_segment);

    record_trace(
        TraceEntry::SEGMENT_FREE,
        reinterpret_cast<size_t>(block->ptr),
        block->size,
        block->queue,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_segment_allocated);

    auto* pool = block->pool;
    deletePrimitive(block->ptr, pool);

    if (pool->owner_PrivatePool) {
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->allocation_count > 0);
      pool->owner_PrivatePool->allocation_count--;
    }
    pool->blocks.erase(block);

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.segment[stat_type].decrease(1);
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

    delete block;
  }

  void unmap_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    auto unmapped =
        block->expandable_segment->unmap(SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size = unmapped.ptr - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // If the actual unmapped region starts after block->ptr due to alignment,
      // the region before unmapped.ptr is still mapped.
      // [Prev Block?] -> [Before Block] -> [Unmapped Block]
      Block* before_free = new Block(
          block->device, block->queue, before_size, block->pool, block->ptr);
      before_free->expandable_segment = block->expandable_segment;
      before_free->splice(block->prev, block);
      block->pool->blocks.insert(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // If the actual unmapped region ends before block->ptr + block->size,
      // the region after (unmapped.ptr + unmapped.size) is still mapped.
      // [Unmapped Block] -> [After Block] -> [Next Block?]
      Block* after_free = new Block(
          block->device,
          block->queue,
          after_size,
          block->pool,
          unmapped.ptr + unmapped.size);
      after_free->expandable_segment = block->expandable_segment;
      after_free->splice(block, block->next);
      block->pool->blocks.insert(after_free);
    }

    // [Before Mapped Block?] -> [Unmapped Block] -> [After Mapped Block?]
    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(unmapped.size);
    });

    if (block->pool->owner_PrivatePool) {
      // The Freed block belonged to a XPU graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(
          block->pool->owner_PrivatePool->allocation_count > 0);
      block->pool->owner_PrivatePool->allocation_count--;
    }

    record_trace(
        TraceEntry::SEGMENT_UNMAP,
        reinterpret_cast<size_t>(unmapped.ptr),
        unmapped.size,
        block->queue,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_segment_allocated);
  }

  void release_blocks(
      BlockPool& pool,
      const std::shared_ptr<GatheredContext>& context) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks in the given pool.
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (block->expandable_segment) {
        // unmap_block() modifies the free pool, so collect items to free first
        // to avoid iterator invalidation.
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block, context);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block, context);
      // After unmap_block(), expandable segment blocks with no neighbors are
      // also released.
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  bool release_cached_blocks(
      const std::shared_ptr<GatheredContext>& context,
      MempoolId_t mempool_id) {
    bool streams_synced = false;
    if (mempool_id.first == 0 && mempool_id.second == 0 &&
        captures_underway.empty()) {
      synchronize_and_free_events(context);
      // See Note [Safe to Free Blocks on BlockPool]
      c10::xpu::syncStreamsOnDevice(device_index);
      streams_synced = true;

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

      if (!streams_synced) {
        // See Note [Safe to Free Blocks on BlockPool]
        c10::xpu::syncStreamsOnDevice(device_index);
        streams_synced = true;
      }
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks, context);
      release_blocks(it->second->large_blocks, context);
      if (it->second->allocation_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }
    return true;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        AcceleratorAllocatorConfig::use_expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return remaining > kSmallSize;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  Block* alloc_found_block(
      const AllocParams& params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    BlockPool* pool = params.pool;
    sycl::queue* queue = params.queue();

    TORCH_INTERNAL_ASSERT(
        params.block != nullptr && params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, queue, size, pool, block->ptr);
      block->expandable_segment = remaining->expandable_segment;
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split && !block->expandable_segment) {
        // Allocate from an existing inactive split block: decrease inactive
        // split bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          stats.inactive_split_bytes[stat_type].decrease(block->size);
        });
      } else if (!block->expandable_segment) {
        // First time split a non-expandable block: create a new inactive
        // split block (the remaining part), so increase the inactive split
        // count and bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          stats.inactive_split[stat_type].increase(1);
          stats.inactive_split_bytes[stat_type].increase(remaining->size);
        });
      }
    } else if (already_split && !block->expandable_segment) {
      // Allocate the whole inactive split block: decrease both count and bytes.
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        stats.inactive_split[stat_type].decrease(1);
        stats.inactive_split_bytes[stat_type].decrease(block->size);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;

    block->context_when_allocated = std::move(context);
    record_trace(
        TraceEntry::ALLOC,
        reinterpret_cast<size_t>(block->ptr),
        orig_size,
        block->queue,
        block->device,
        block->pool->owner_MempoolId(),
        block->context_when_allocated);

    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted)

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      stats.allocation[stat_type].increase(1);
      stats.active[stat_type].increase(1);
      stats.allocated_bytes[stat_type].increase(block->size);
      stats.active_bytes[stat_type].increase(block->size);
      stats.requested_bytes[stat_type].increase(block->requested_size);
    });

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::XPU, device));

    return block;
  }

  void insert_events(Block* block) {
    stream_set streams(std::move(block->stream_uses));
    TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      block->event_count++;
      xpu_events[stream].emplace_back(
          stream.queue().ext_oneapi_submit_barrier(), block);
    }
  }

  std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const {
    std::vector<Block*> blocks;
    for (Block* b : active_blocks) {
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    for (Block* b : pool->small_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }
    for (Block* b : pool->large_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    return blocks;
  }

  void create_or_incref_pool(
      MempoolId_t mempool_id,
      XPUAllocator* allocator = nullptr) {
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool.
      // Make a new pool for XPU graph capture or memory pool usage.
      graph_pools.emplace(
          mempool_id, std::make_unique<PrivatePool>(mempool_id, allocator));
    } else {
      // mempool_id references an existing pool, which the current XPU graph
      // capture will share.
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      TORCH_INTERNAL_ASSERT(allocator == nullptr);
      it->second->use_count++;
    }
  }

  PrivatePool* get_private_pool(MempoolId_t mempool_id) {
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    return it->second.get();
  }

  // This function must be called with the mutex unlocked. Holding the mutex
  // here can cause deadlocks, especially when interacting with Python code that
  // may acquire the GIL or invoke callbacks.
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

 public:
  DeviceCachingAllocator(DeviceIndex device_index)
      : large_blocks(/* small */ false),
        small_blocks(/* small */ true),
        device_index(device_index) {}

  Block* malloc(DeviceIndex device, size_t orig_size, sycl::queue& queue) {
    // Outside mutex to avoid deadlocks.
    auto context = maybeGatherContext(RecordContext::STATE);

    std::scoped_lock<std::recursive_mutex> lock(mutex);
    if (C10_LIKELY(captures_underway.empty())) {
      process_events(context);
    }
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, &queue);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, &queue, &pool, alloc_size);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);
    // Can't reuse an existing block, try to get a new one.
    if (!block_found) {
      block_found = alloc_block(params, false, context) ||
          (release_cached_blocks(context, {0, 0}) &&
           alloc_block(params, true, context));
    }
    if (!block_found) {
      const auto& raw_device = c10::xpu::get_raw_device(device);
      const auto device_total =
          raw_device.get_info<sycl::info::device::global_mem_size>();
      // Estimate the available device memory when the SYCL runtime does not
      // support the corresponding aspect (ext_intel_free_memory).
      size_t device_free = device_total -
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      // TODO: Remove the aspect check once the SYCL runtime bug is fixed on
      // affected devices.
      if (raw_device.has(sycl::aspect::ext_intel_free_memory)) {
        device_free =
            raw_device.get_info<sycl::ext::intel::info::device::free_memory>();
      }
      std::string allowed_info;
      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;

      record_trace(
          TraceEntry::OOM,
          device_free,
          params.size(),
          params.queue(),
          params.device(),
          params.pool->owner_MempoolId(),
          std::move(context));

      c10::reportOutOfMemoryToProfiler(
          static_cast<int64_t>(size),
          allocated_bytes,
          reserved_bytes,
          c10::Device(c10::DeviceType::XPU, device));

      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "XPU out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". GPU ",
          static_cast<int>(device),
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. ",
          allowed_info,
          "Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " Please use `empty_cache` to release all unoccupied cached memory.");
    }
    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        params, orig_size, std::move(context), split_remainder);
  }

  void free(Block* block) {
    // Outside mutex to avoid deadlocks.
    auto context = maybeGatherContext(RecordContext::ALL);

    std::scoped_lock<std::recursive_mutex> lock(mutex);
    block->allocated = false;

    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocation[stat_type].decrease(1);
      stats.allocated_bytes[stat_type].decrease(block->size);
    });

    record_trace(
        TraceEntry::FREE_REQUESTED,
        reinterpret_cast<size_t>(block->ptr),
        block->requested_size,
        block->queue,
        block->device,
        block->pool->owner_MempoolId(),
        context ? context : block->context_when_allocated);

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block, context);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -static_cast<int64_t>(orig_block_size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::XPU, block->device));
  }

  void recordStream(Block* block, xpu::XPUStream stream) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    if (stream.queue() == *block->queue) {
      return;
    }
    block->stream_uses.insert(stream);
  }

  void emptyCache(MempoolId_t mempool_id) {
    // Outside mutex to avoid deadlocks.
    auto context = maybeGatherContext(RecordContext::ALL);

    std::scoped_lock<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context, mempool_id);
  }

  DeviceStats getStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void resetAccumulatedStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

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
  }

  void resetPeakStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

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
  }

  void record_trace(
      TraceEntry::Action action,
      size_t addr,
      size_t size,
      sycl::queue* queue,
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::shared_ptr<GatheredContext> context) {
    if (!record_history)
      return;
    bool should_skip = skip_actions_list.count(action) > 0;
    if (should_skip)
      return;
    TraceEntry te(
        action,
        device,
        addr,
        size,
        reinterpret_cast<void*>(queue),
        mempool_id,
        getApproximateTime(),
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);
    alloc_buffer.insertEntries(te);
  }

  std::vector<SegmentInfo> snapshot(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<Block*> all_blocks;

    if (mempool_id.first != 0 || mempool_id.second != 0) {
      // If there is an active mempool, we find the corresponding PrivatePool
      // in graph_pools and only return the blocks from it.
      auto pool = graph_pools.find(mempool_id);
      if (pool != graph_pools.end()) {
        all_blocks = get_private_pool_head_blocks(pool->second.get());
      }
    } else {
      // When snapshot is called with non-default mempool_id, we return
      // all the blocks from all pools.
      all_blocks = get_all_blocks();
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;

    for (const Block* const head_block : all_blocks) {
      // For expandable segments, we report one segment for each contiguous
      // mapped range of memory
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<size_t>(head_block->ptr);
      segment_info.stream = reinterpret_cast<void*>(head_block->queue);
      segment_info.is_large = (!head_block->pool->is_small);
      segment_info.is_expandable = head_block->expandable_segment;
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;
      MempoolId_t id = head_block->pool->owner_MempoolId();
      if ((mempool_id.first == 0 && mempool_id.second == 0) ||
          id == mempool_id) {
        segment_info.owner_private_pool_id = id;
      }

      const Block* block = head_block;
      while (block != nullptr && block->mapped) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        block_info.context_when_allocated = block->context_when_allocated;
        block = block->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    record_trace(
        TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, mempool_id, nullptr);
    return result;
  }

  std::vector<TraceEntry> trace(
      const std::function<time_t(approx_time_t)>& tsc_to_us) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    alloc_buffer.getEntries(result);

    // Convert all the timestamps from tsc to epoch time in microseconds.
    for (auto& te : result) {
      te.time_.t_ = tsc_to_us(te.time_.approx_t_);
    }
    return result;
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_buffer_max_entries,
      RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;

    skip_actions_list.clear();
    for (const auto& action_str : skip_actions) {
      auto action = parseTraceEntryAction(action_str);
      skip_actions_list.insert(action);
    }

    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_buffer.setMaxEntries(alloc_buffer_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    if (!enabled || clearHistory) {
      alloc_buffer.clear();
    }
  }

  std::pair<size_t, size_t> getMemoryInfo() {
    const auto& device = c10::xpu::get_raw_device(device_index);
    const size_t total = device.get_info<sycl::info::device::global_mem_size>();
    TORCH_CHECK(
        device.has(sycl::aspect::ext_intel_free_memory),
        "The device (",
        device.get_info<sycl::info::device::name>(),
        ") doesn't support querying the available free memory. ",
        "You can file an issue at https://github.com/pytorch/pytorch/issues ",
        "to help us prioritize its implementation.");
    const size_t free =
        device.get_info<sycl::ext::intel::info::device::free_memory>();
    return {free, total};
  }

  double getMemoryFraction() {
    if (!set_fraction) {
      return 1.0;
    }

    const auto device_total =
        xpu::get_raw_device(device_index)
            .get_info<sycl::info::device::global_mem_size>();
    return static_cast<double>(allowed_memory_maximum) /
        static_cast<double>(device_total);
  }

  void setMemoryFraction(double fraction) {
    const auto device_total =
        xpu::get_raw_device(device_index)
            .get_info<sycl::info::device::global_mem_size>();
    allowed_memory_maximum =
        static_cast<size_t>(fraction * static_cast<double>(device_total));
    set_fraction = true;
  }

  void createOrIncrefPool(
      MempoolId_t mempool_id,
      XPUAllocator* allocator = nullptr) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id, allocator);
  }

  int getPoolUseCount(MempoolId_t mempool_id) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      return 0;
    }
    return it->second->use_count;
  }

  // Called by XPUGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(sycl::queue*)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id);
    auto not_found = std::all_of(
        captures_underway.begin(),
        captures_underway.end(),
        [&](const auto& entry) { return entry.first != mempool_id; });
    TORCH_CHECK(
        not_found, "beginAllocateToPool: already recording to mempool_id");
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by XPUGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto it = std::find_if(
        captures_underway.begin(),
        captures_underway.end(),
        [&](const auto& entry) { return entry.first == mempool_id; });
    TORCH_INTERNAL_ASSERT(
        it != captures_underway.end(),
        "endAllocatePool: not currently recording to mempool_id");
    captures_underway.erase(it);
  }

  // Called by XPUGraph::reset and MemPool::~MemPool()
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto pp = get_private_pool(mempool_id);
    auto uc = --(pp->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      bool inserted = graph_pools_freeable.insert({mempool_id, pp}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }
};

static void local_raw_delete(void* ptr);

class NativeCachingAllocator : public XPUAllocator {
 private:
  alignas(hardware_destructive_interference_size) std::mutex mutex;
  ska::flat_hash_map<void*, Block*> allocated_blocks;
  c10::ApproximateClockToUnixTimeConverter clock_converter;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::scoped_lock<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void assertValidDevice(DeviceIndex device) {
    const auto device_num = device_allocators.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  void init(DeviceIndex device_count) override {
    const auto size = static_cast<DeviceIndex>(device_allocators.size());
    if (size < device_count) {
      device_allocators.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocators[i] = std::make_unique<DeviceCachingAllocator>(i);
      }
    }
  }

  bool initialized() override {
    return !device_allocators.empty();
  }

  void malloc(
      void** devPtr,
      DeviceIndex device,
      size_t size,
      sycl::queue& queue) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocators.size(),
        "Allocator not initialized for device ",
        static_cast<int16_t>(device),
        ": did you call init?");
    Block* block = device_allocators[device]->malloc(device, size, queue);
    add_allocated_block(block);
    *devPtr = block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, /* remove */ true);
    TORCH_CHECK(block, "invalid device pointer: ", ptr);
    device_allocators[block->device]->free(block);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(block->ptr));
    }
  }

  void emptyCache(MempoolId_t mempool_id) override {
    for (auto& da : device_allocators) {
      da->emptyCache(mempool_id);
    }
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    c10::xpu::XPUStream xpu_stream{stream};
    device_allocators[block->device]->recordStream(block, xpu_stream);
  }

  DataPtr allocate(size_t size) override {
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    if (size != 0) {
      this->malloc(&r, device, size, xpu::getCurrentXPUStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::XPU, device)};
  }

  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  void* raw_alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    malloc(&r, device, size, xpu::getCurrentXPUStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t size, XPUStream stream) {
    if (size == 0) {
      return nullptr;
    }
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    malloc(&r, device, size, stream);
    return r;
  }

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
  }

  DeviceStats getDeviceStats(DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocators[device]->getStats();
  }

  void resetPeakStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetPeakStats();
  }

  void resetAccumulatedStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetAccumulatedStats();
  }

  SnapshotInfo snapshot(MempoolId_t mempool_id) {
    // Set-up converter to convert timestamps from tsc to microseconds.
    auto tsc_to_ns = clock_converter.makeConverter();
    auto tsc_to_us = [=](approx_time_t t_approx) {
      return tsc_to_ns(t_approx) / 1000;
    };

    SnapshotInfo result;

    // Get the device_traces' TraceEntry lists.
    for (auto& da : device_allocators) {
      result.device_traces.emplace_back(da->trace(tsc_to_us));
      auto snap = da->snapshot(mempool_id);
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }

    auto& md = result.config_metadata;
    md.expandable_segments =
        AcceleratorAllocatorConfig::use_expandable_segments();
    md.last_allocator_settings =
        AcceleratorAllocatorConfig::last_allocator_settings();
    return result;
  }

  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
    assertValidDevice(dev);
    assertValidDevice(dev_to_access);
    c10::xpu::get_raw_device(dev).ext_oneapi_enable_peer_access(
        c10::xpu::get_raw_device(dev_to_access));
  }

  std::pair<size_t, size_t> getMemoryInfo(DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocators[device]->getMemoryInfo();
  }

  double getMemoryFraction(DeviceIndex device) {
    assertValidDevice(device);
    return device_allocators[device]->getMemoryFraction();
  }

  void setMemoryFraction(double fraction, DeviceIndex device) {
    assertValidDevice(device);
    TORCH_CHECK_VALUE(
        0 < fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1].");
    device_allocators[device]->setMemoryFraction(fraction);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_buffer_max_entries,
      RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) {
    for (auto& allocator : device_allocators) {
      allocator->recordHistory(
          enabled,
          context_recorder,
          alloc_buffer_max_entries,
          when,
          clearHistory,
          skip_actions);
    }
  }

  void createOrIncrefPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      XPUAllocator* allocator) {
    assertValidDevice(device);
    device_allocators[device]->createOrIncrefPool(
        std::move(mempool_id), allocator);
  }

  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(sycl::queue*)> filter) {
    assertValidDevice(device);
    device_allocators[device]->beginAllocateToPool(
        std::move(mempool_id), std::move(filter));
  }

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocators[device]->endAllocateToPool(mempool_id);
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocators[device]->releasePool(std::move(mempool_id));
  }

  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    return device_allocators[device]->getPoolUseCount(std::move(mempool_id));
  }
};

static NativeCachingAllocator native_allocator;

void local_raw_delete(void* ptr) {
  native_allocator.free(ptr);
}

std::atomic<XPUAllocator*> allocator;

struct NativeAllocatorStaticInitializer {
  NativeAllocatorStaticInitializer() {
    allocator.store(&native_allocator);
    c10::SetAllocator(c10::kXPU, &native_allocator, 0);
  }
};

static NativeAllocatorStaticInitializer native_allocator_static_initializer;

void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  return native_allocator.enablePeerAccess(dev, dev_to_access);
}

double getMemoryFraction(DeviceIndex device) {
  return native_allocator.getMemoryFraction(device);
}

void setMemoryFraction(double fraction, DeviceIndex device) {
  return native_allocator.setMemoryFraction(fraction, device);
}

void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions) {
  native_allocator.recordHistory(
      enabled,
      context_recorder,
      alloc_trace_max_entries,
      when,
      clearHistory,
      skip_actions);
}

SnapshotInfo snapshot(MempoolId_t mempool_id) {
  return native_allocator.snapshot(mempool_id);
}

void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    XPUAllocator* allocator_ptr) {
  return native_allocator.createOrIncrefPool(device, mempool_id, allocator_ptr);
}

void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(sycl::queue*)> filter) {
  return native_allocator.beginAllocateToPool(
      device, mempool_id, std::move(filter));
}

void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return native_allocator.endAllocateToPool(device, mempool_id);
}

void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return native_allocator.releasePool(device, mempool_id);
}

int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return native_allocator.getPoolUseCount(device, mempool_id);
}

} // namespace c10::xpu::XPUCachingAllocator

namespace c10::xpu {

// uid_ is incremented when a user creates a MemPool,
//
// uuid_ is incremented when XPUGraph creates a MemPool
// as a result of a user not providing a pool.

std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    XPUCachingAllocator::XPUAllocator* allocator,
    bool is_user_created,
    bool use_on_oom)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::xpu::current_device();
  XPUCachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    // XPU doesn't support use_on_oom yet
    TORCH_WARN(
        "XPUCachingAllocator::MemPool: use_on_oom is not supported on XPU");
  }
}

MemPool::~MemPool() {
  TORCH_INTERNAL_ASSERT(use_count() == 1);
  XPUCachingAllocator::releasePool(device_, id_);
  c10::xpu::XPUCachingAllocator::emptyCache(id_); // release cached blocks
}

MempoolId_t MemPool::id() {
  return id_;
}

XPUCachingAllocator::XPUAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return XPUCachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace c10::xpu
