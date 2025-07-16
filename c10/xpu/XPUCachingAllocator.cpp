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

XPUAllocatorConfig::XPUAllocatorConfig()
    : m_expandable_segments(false) {}

void XPUAllocatorConfig::lexArgs(
    const std::string& env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  for (char ch : env) {
    if (ch == ',' || ch == ':' || ch == '[' || ch == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, ch);
    } else if (ch != ' ') {
      buf.emplace_back(ch);
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void XPUAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c),
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

void XPUAllocatorConfig::parseArgs(const std::optional<std::string>& env) {
  // If empty, set the default values
  if (!env.has_value()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(m_last_allocator_settings_mutex);
    m_last_allocator_settings = env.value();
  }

  std::vector<std::string> config;
  lexArgs(env.value(), config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "expandable_segments") {
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for expandable_segments");
      config_item_view = config[i];
      m_expandable_segments = (config_item_view == "True");
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  XPUCachingAllocator::XPUAllocatorConfig::instance().parseArgs(env.c_str());
}

// newly allocated memory with 512-byte alignment.
constexpr size_t kDeviceAlignment = 512;
// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize = 512;
// largest "small" allocation is 1 MiB
constexpr size_t kSmallSize = 1048576;
// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152;
// "large" allocations may be packed in 20 MiB blocks
constexpr size_t kLargeBuffer = 20971520;
// allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 2097152;

namespace {
using stream_set = ska::flat_hash_set<xpu::XPUStream>;

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
bool BlockComparatorSize(const Block* a, const Block* b);
bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool {
  BlockPool(bool small) : blocks(BlockComparatorSize), unmapped(BlockComparatorAddress), is_small(small) {}
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  const bool is_small;

  // Add a Block into blocks set with updating gc counter.
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(
      Block* block);
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
  bool mapped{true}; // is the virtual address range this Block references
                     // backed by physical pages. Always true when
                     // expandable_segment_ is null. When false
                     // This Block will be aligned to the segment size
                     // of its expandable_segment_.
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding XPU events
  ExpandableSegment* expandable_segment_{nullptr};

  Block(
      DeviceIndex device,
      sycl::queue* queue,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(DeviceIndex device, sycl::queue* queue, size_t size)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
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

std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::
    insert_into_blocks(Block* block) {
  return blocks.insert(block);
}

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};


struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<sycl::queue*> queue,
      size_t address_space_size,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers)
      : device_(device),
        queue_(queue),
        // 2MB for small pool, 20MB for large pool
        segment_size_(segment_size),
        max_handles_(numSegments(address_space_size)),
        peers_(std::move(peers)) {
    c10::xpu::DeviceProp prop;
    c10::xpu::get_device_properties(&prop, device_);
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.global_mem_size + prop.global_mem_size / 8);
    ptr_ = sycl::ext::oneapi::experimental::reserve_virtual_mem(segment_size_ * max_handles_, (*queue_)->get_context());
  }
  ExpandableSegment(const ExpandableSegment&) = delete;
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment operator=(const ExpandableSegment&) = delete;
  ExpandableSegment operator=(ExpandableSegment&&) = delete;

  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  // return nullptr indicates the handle type is not supported.
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }

    while (end > handles_.size()) {
      handles_.emplace_back(std::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      sycl::ext::oneapi::experimental::physical_mem* handle;
      try {
        handle = new sycl::ext::oneapi::experimental::physical_mem((*queue_)->get_device(), (*queue_)->get_context(), segment_size_);
      } catch (const sycl::exception& e) {
        if (e.code() == sycl::errc::memory_allocation) {
          for (auto j : c10::irange(begin, i)) {
            auto h = handles_.at(j).value();
            handles_.at(j) = std::nullopt;
            delete h.handle;
          }
          trimHandles();
          return rangeFromHandles(begin, begin);
        }else {
          return SegmentRange(nullptr, 0);
        }
      }
      handles_.at(i) = Handle{handle};
    }
    mapAndSetAccess(begin, end);
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const {
    return reinterpret_cast<char*>(ptr_);
  }

  size_t size() const {
    return max_handles_ * segment_size_;
  }

  void addPeer(c10::DeviceIndex device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    sycl::ext::oneapi::experimental::free_virtual_mem(ptr_, segment_size_ * max_handles_, (*queue_)->get_context());
  }

 private:
  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) {
    sycl::ext::oneapi::experimental::set_access_mode((void*)(ptr_ + begin * segment_size_), (end - begin) * segment_size_, sycl::ext::oneapi::experimental::address_access_mode::read_write, (*queue_)->get_context());
  }

  void mapAndSetAccess(size_t begin, size_t end) {
    for (auto i : c10::irange(begin, end)) {
      handles_.at(i).value().handle->map(ptr_ + i * segment_size_, segment_size_, sycl::ext::oneapi::experimental::address_access_mode::read_write);
    }
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
  }

  void unmapHandles(size_t begin, size_t end) {
    if(queue_){
      (*queue_)->wait(); // wait for the stream to finish before unmapping
    }else {
      c10::xpu::syncStreamsOnDevice(device_);
    }
    for (auto i : c10::irange(begin, end)) {
      Handle h = handles_.at(i).value();
      handles_.at(i) = std::nullopt;
      sycl::ext::oneapi::experimental::unmap(reinterpret_cast<void *>(ptr_ + segment_size_ * i), segment_size_, (*queue_)->get_context());
      delete h.handle;
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
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
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }
  c10::DeviceIndex device_;
  std::optional<sycl::queue*> queue_;
  uintptr_t ptr_;
  size_t segment_size_;
  size_t max_handles_;
  struct Handle {
    sycl::ext::oneapi::experimental::physical_mem* handle;
  };
  std::vector<std::optional<Handle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<c10::DeviceIndex> peers_;
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

struct AllocParams {
  AllocParams(
      DeviceIndex device,
      size_t size,
      sycl::queue* queue,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, queue, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr) {}

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
  Block* block;
  StatTypes stat_types = {};
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

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

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

  std::vector<c10::DeviceIndex> peers() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return devices_with_peer_access_;
  }

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
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
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    c10::xpu::DeviceProp prop;
    c10::xpu::get_device_properties(&prop, device);
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    size_t address_space_size = prop.global_mem_size + prop.global_mem_size / 8;

    expandable_segments_.emplace_back(new ExpandableSegment(
        device,
        queue,
        address_space_size,
        segment_size,
        devices_with_peer_access_));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, queue, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(
      Block* to_map,
      size_t size){
      // const std::shared_ptr<GatheredContext>& ctx) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    // TORCH_INTERNAL_ASSERT(
    //     !to_map->context_when_allocated); // unmapped blocks should not keep
    //                                       // history
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
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
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.insert_into_blocks(to_map);

    // update statistics
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(mapped_range.size);
    });
    return true;
  }

  Block* try_allocate_expandable_block(
      c10::DeviceIndex device,
      sycl::queue* queue,
      BlockPool* pool,
      size_t size){
      // const std::shared_ptr<GatheredContext>& ctx) {
    Block* candidate = find_expandable_block(device, queue, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size))) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(
              new_candidate, std::min(remaining, candidate->next->size))) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }

  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;
    auto& pool = *block->pool;
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      try_merge_blocks(block, merge_candidate, pool);
    }

    active_blocks.erase(block);
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segments from
      // inactive_split
      if (!block->expandable_segment_) {
        stats.active_bytes[stat_type].decrease(original_block_size);
        stats.requested_bytes[stat_type].decrease(requested_size);
      }
    });
  }

  void process_events() {
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
          free_block(block);
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
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  BlockPool& get_pool(size_t size) {
    if (size < kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
      return false;
    }

    if ((*it)->expandable_segment_) {
      if (XPUAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](Block* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->queue == p.queue() &&
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

  bool alloc_block(AllocParams& p, bool isRetry) {
    auto size = p.alloc_size;
    auto device = p.device();
    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
    if (XPUAllocatorConfig::expandable_segments()) {
      p.block = try_allocate_expandable_block(
          p.device(), p.queue(), p.pool, p.size());
      return bool(p.block);
    }
    void* ptr = sycl::aligned_alloc_device(
        kDeviceAlignment,
        size,
        xpu::get_raw_device(device),
        xpu::get_device_context());
    if (!ptr) {
      return false;
    }
    p.block = new Block(device, p.queue(), size, p.pool, ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(size);
    });
    return true;
  }

  void synchronize_and_free_events() {
    for (auto& xe : xpu_events) {
      for (auto& e : xe.second) {
        auto event = e.first;
        auto* block = e.second;
        event.wait();
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
      }
    }
    xpu_events.clear();
  }

  void release_expandable_segment(Block* block) {
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

  void release_block(Block* block) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * Callers must ensure that all accesses to the block, whose raw pointer is
     * allocated by SYCL APIs, have been completed before invoking sycl::free.
     *
     * We have to do a device-level synchronization before free these blocks to
     * guarantee that all kernels can access to the blocks have finished.
     */
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    sycl::free(block->ptr, xpu::get_device_context());
    auto* pool = block->pool;
    pool->blocks.erase(block);

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

    delete block;
  }

  void unmap_block(Block* block) {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size =
        static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      Block* before_free = new Block(
          block->device, block->queue, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->insert_into_blocks(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block* after_free = new Block(
          block->device,
          block->queue,
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

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    // update statistics
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(unmapped.size);
    });
  }

  void release_blocks(BlockPool& pool) {
    std::vector<Block*> to_unmap;
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  bool release_cached_blocks() {
    synchronize_and_free_events();
    // See Note [Safe to Free Blocks on BlockPool]
    c10::xpu::syncStreamsOnDevice(device_index);

    release_blocks(large_blocks);
    release_blocks(small_blocks);
    return true;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small || XPUAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return remaining > kSmallSize;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    BlockPool* pool = params.pool;
    sycl::queue* queue = params.queue();

    TORCH_INTERNAL_ASSERT(
        params.block != nullptr && params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    if (split_remainder) {
      remaining = block;

      block = new Block(device, queue, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
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
    }

    block->allocated = true;
    block->requested_size = orig_size;
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted)

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
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

 public:
  DeviceCachingAllocator(DeviceIndex device_index)
      : large_blocks(/* small */ false),
        small_blocks(/* small */ true),
        device_index(device_index) {}

  Block* malloc(DeviceIndex device, size_t orig_size, sycl::queue& queue) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    process_events();
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, &queue, &pool, alloc_size);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);
    // Can't reuse an existing block, try to get a new one.
    if (!block_found) {
      block_found = alloc_block(params, false) ||
          (release_cached_blocks() && alloc_block(params, true));
    }
    if (!block_found) {
      c10::xpu::DeviceProp device_prop;
      c10::xpu::get_device_properties(&device_prop, device);
      auto device_total = device_prop.global_mem_size;
      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;

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
          ". Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " Please use `empty_cache` to release all unoccupied cached memory.");
    }
    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(std::move(params), orig_size, split_remainder);
  }

  void free(Block* block) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    block->allocated = false;

    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocated_bytes[stat_type].decrease(block->size);
    });

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
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

  void emptyCache() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
  }

  DeviceStats getStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void resetAccumulatedStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_accumulated();
      stats.reserved_bytes[statType].reset_accumulated();
      stats.active_bytes[statType].reset_accumulated();
      stats.requested_bytes[statType].reset_accumulated();
    }
    stats.num_alloc_retries = 0;
  }

  void resetPeakStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_peak();
      stats.reserved_bytes[statType].reset_peak();
      stats.active_bytes[statType].reset_peak();
      stats.requested_bytes[statType].reset_peak();
    }
  }
};

static void local_raw_delete(void* ptr);

class XPUAllocator : public Allocator {
 private:
  std::mutex mutex;
  ska::flat_hash_map<void*, Block*> allocated_blocks;

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

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  void init(DeviceIndex device_count) {
    const auto size = static_cast<DeviceIndex>(device_allocators.size());
    if (size < device_count) {
      device_allocators.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocators[i] = std::make_unique<DeviceCachingAllocator>(i);
      }
    }
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

  void emptyCache() {
    for (auto& da : device_allocators) {
      da->emptyCache();
    }
  }

  void recordStream(const DataPtr& ptr, XPUStream stream) {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    device_allocators[block->device]->recordStream(block, stream);
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

  void* raw_alloc(size_t size) {
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

  void raw_delete(void* ptr) {
    this->free(ptr);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
  }

  void assertValidDevice(DeviceIndex device) {
    const auto device_num = device_allocators.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  DeviceStats getDeviceStats(DeviceIndex device) {
    assertValidDevice(device);
    return device_allocators[device]->getStats();
  }

  void resetPeakStats(DeviceIndex device) {
    assertValidDevice(device);
    device_allocators[device]->resetPeakStats();
  }

  void resetAccumulatedStats(DeviceIndex device) {
    assertValidDevice(device);
    device_allocators[device]->resetAccumulatedStats();
  }
};

static XPUAllocator allocator;

void local_raw_delete(void* ptr) {
  allocator.free(ptr);
}

Allocator* get() {
  return &allocator;
}

void init(DeviceIndex device_count) {
  return allocator.init(device_count);
}

void emptyCache() {
  return allocator.emptyCache();
}

void resetPeakStats(DeviceIndex device) {
  return allocator.resetPeakStats(device);
}

void resetAccumulatedStats(DeviceIndex device) {
  return allocator.resetAccumulatedStats(device);
}

DeviceStats getDeviceStats(DeviceIndex device) {
  return allocator.getDeviceStats(device);
}

void* raw_alloc(size_t size) {
  return allocator.raw_alloc(size);
}

void raw_delete(void* ptr) {
  return allocator.raw_delete(ptr);
}

void recordStream(const DataPtr& dataPtr, XPUStream stream) {
  return allocator.recordStream(dataPtr, stream);
}

REGISTER_ALLOCATOR(kXPU, &allocator)

} // namespace c10::xpu::XPUCachingAllocator
