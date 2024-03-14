#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/static_tracepoint.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <utility>
#include <vector>

TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda::CUDACachingAllocator {

// Included here as this is externally used in CUDAAllocatorConfig
const size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks

namespace Native {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB

namespace {

using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void increase_stat(Stat& stat, size_t amount) {
  stat.current += static_cast<int64_t>(amount);
  stat.peak = std::max(stat.current, stat.peak);
  stat.allocated += static_cast<int64_t>(amount);
}

void decrease_stat(Stat& stat, size_t amount) {
  stat.current -= static_cast<int64_t>(amount);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");
  stat.freed += static_cast<int64_t>(amount);
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void decrease_stat_array(
    StatArray& stat_array,
    size_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        decrease_stat(stat_array[stat_type], amount);
      });
}

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
  int64_t get_free_blocks_call_count{0};

  // Add a Block into blocks set with updating gc counter.
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(
      Block* block);
};

struct ExpandableSegment;

struct Block {
  c10::DeviceIndex device; // gpu
  cudaStream_t stream; // allocation stream
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
  int event_count{0}; // number of outstanding CUDA events
  int64_t gc_count_base{0}; // get_free_blocks_call_count when Block is inserted
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};

  Block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr),
        gc_count_base(0) {}

  // constructor for search key
  Block(c10::DeviceIndex device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0) {}

  size_t gc_count() {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

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
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

/*
Note [Expandable Segments]

Rationale

For large (>2MB) allocations, the allocator calls cudaMalloc to get allocations
that are the same size as what the user requests. In the future, parts of these
allocations can be reused for other requests if they are free. This works well
when the program makes many requests of exactly the same size or of sizes that
even multiples of that size. Many deep learning models follow this behavior.
However, one common exception is when the batch size changes slightly from one
iteration to the next, e.g. in batched inference. When the program runs
initially with batch size N, it will make allocations appropriate for that size.
If in the future, it runs at size N - 1, the existing allocations will still be
big enough. However, if it runs at size N + 1, then it will have to make new
allocations that are slightly larger. Not all the tensors are the same size.
Some might be (N + 1)*A and others (N + 1)*A*B where A and B are some non-batch
dimensions in the model. Because the allocator reuses existing allocations when
they are big enough, some number of (N + 1)*A allocations will actually fit in
the already existing N*B*A segments, though not perfectly. As the model runs it
will partially fill up all of these segments leaving unusable free slices of
memory at the end of these segments. The allocator at some point will need to
cudaMalloc a new (N + 1)*A*B segment. If there is not enough memory, there is
now no way to recover the slices of memory that are free at the end of existing
segments. With models 50+ layers deep, this pattern might repeat 50+ times
creating many slivers.

Approach

Expandable segments allows the allocator to create a segment initially and then
expand its size later when more memory is needed. Instead of making one segment
per allocation, it tries to make one segment (per stream) that grows as
necessary. Now when the N + 1 case runs, the allocations will tile nicely into
the one large segment until it fills up. Then more memory is requested and
appended to the end of the segment. This process does not create as many slivers
of unusable memory, so it is more likely to succeed at finding this memory.

Implementation

The expandable_segments:True option is used to enable/disable this behavior. We
use cuda's low-level memory APIs, which are similar to mmap, to extend the
memory segments. These APIs separate the allocation of physical memory
(cuMemCreate) from the allocation of virtual address space (cuMemAddressReserve)
and the associate between them cuMemMap/cuMemSetAccess.

When we allocate a new segment, we allocate enough address space to map
basically the entire physical memory of the GPU (there is 256TiB of address
space), but we only map enough physical memory to handle the current amount of
memory needed by the program. As more is requested, we add more physical memory
to the segment. This can work at the granularity of GPU pages which are 2MiB
currently.

If we end up out of memory, we can unmap all the memory in our segment
corresponding to empty physical pages, and return it to CUDA for use at another
address in the segment or in a segment for a different stream.

A current limitation of CUDA's API is that physical memory
(CUmemGenericAllocationHandle) cannot be split up after it is mapped even if the
handle holds multiple GPU pages. The cost to map/unmap memory is proportional to
the number of physical memory chunks that were allocated (mapping 10 separately
allocated 2MiB pages takes 10x time compared to mapping one 20MiB physical
allocation of 10 pages).  Changing memory mappings also appears to involve at
least some synchronous actions with the GPU and so should be considered an
expensive operation. To limit overhead, we use 2MiB pages for our small pool and
20MiB pages for our large pool. Initially allocation using expandable_blocks
will be slower than cudaMalloc, though still in the milliseconds range for
mapping the entire memory.

When mapping new memory to expand the segment, we look for the lowest address at
which we can fit a new allocation by adding new pages. Normally this will be at
the end of the block. But if have previously unmapped blocks earlier in the
segment during an OOM, it will first try to fill in those gaps to keep the
segment as a single block. By allocating at the lowest address we encourage
the split up parts of the block to merge into a single block again, reducing
fragmentation potential.

Allocation of blocks in the segment uses the same best-fit heuristics of the
rest of the allocator.

Expandable blocks can be enabled/disabled throughout the run of a program. When
disabled, the allocator will not put new allocations in an expandable block.

Limitations

* Slightly slower initial memory allocation speed.
* IPC of cuda tensors (e.g. for multiprocess dataloaders) is not supported.
However, it is possible to temporarily disable (expandable_segments:False) the
bevhavior for allocator tensors that need to be used cross-process.
* CUDA runtime APIs related to sharing memory across process
(cudaDeviceEnablePeerAccess) do not work for memory allocated with cuMemMap.
Instead these mapping have to be done manually. The allocator now has an
`enablePeerAccess` method to do this.
*/

struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      std::vector<c10::DeviceIndex> peers)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size),
        peers_(std::move(peers)) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
        &ptr_, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
  }
  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      CUmemGenericAllocationHandle handle = 0;
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = static_cast<int>(device_);
      auto status =
          DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);
      if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      C10_CUDA_DRIVER_CHECK(status);
      handles_.at(i) = handle;
    }
    for (auto i : c10::irange(begin, end)) {
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          handles_.at(i).value(),
          0ULL));
    }

    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
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
    return (char*)ptr_;
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
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
        ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = static_cast<int>(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (auto i : c10::irange(begin, end)) {
      CUmemGenericAllocationHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          ptr_ + segment_size_ * i, segment_size_));
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
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
  cudaStream_t stream_;
  CUdeviceptr ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<CUmemGenericAllocationHandle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<c10::DeviceIndex> peers_;
};
#else
struct ExpandableSegment {
  ExpandableSegment(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      const std::vector<c10::DeviceIndex>& peers) {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  char* ptr() const {
    return nullptr;
  }
  size_t size() const {
    return 0;
  }
  void addPeer(c10::DeviceIndex device) {}
};
#endif

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  c10::DeviceIndex device = 0;
  cudaStream_t stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int gc_count_base = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cudaMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

static bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}
static bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(cudaSuccess) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  cudaError_t err;
};

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
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
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
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
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA graphs helper
struct PrivatePool {
  PrivatePool()
      : use_count(1),
        cudaMalloc_count(0),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // Number of live graphs using this pool
  int use_count;
  // Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critical though,
  // I'd rather not add more logic to it.
  BlockPool large_blocks;
  BlockPool small_blocks;
};

BlockState::BlockState(Block* block)
    : stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count_base(block->gc_count_base) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
};

SegmentState::SegmentState(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  is_small = head->pool->is_small;

  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id)) {
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
#endif
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  } else {
    // It's ok to capture cudaMallocs, as long as we never cudaFree those
    // addresses before replay.
    // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  }
#endif
}

} // anonymous namespace
} // namespace Native

static std::string reportProcessMemoryInfo(c10::DeviceIndex device) {
#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  void* nvml_handle = DriverAPI::get_nvml_handle();
  if (!nvml_handle) {
    return "";
  }
  static c10::once_flag nvml_init;
  c10::call_once(nvml_init, [] {
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
  });

  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  char pci_id[80];
  snprintf(
      pci_id,
      sizeof(pci_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  nvmlDevice_t nvml_device = nullptr;
  TORCH_INTERNAL_ASSERT(
      NVML_SUCCESS ==
      DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
          pci_id, &nvml_device));

  std::vector<nvmlProcessInfo_v1_t> procs(8);
  unsigned int size = procs.size();
  nvmlReturn_t r;
  while ((r = DriverAPI::get()->nvmlDeviceGetComputeRunningProcesses_(
              nvml_device, &size, procs.data())) ==
         NVML_ERROR_INSUFFICIENT_SIZE) {
    procs.resize(size);
  }
  unsigned int self_pid = getpid();
  std::stringstream ss;
  TORCH_INTERNAL_ASSERT(NVML_SUCCESS == r);
  ss << "";
  for (auto i : c10::irange(size)) {
    auto& proc = procs[i];
    if (self_pid == proc.pid) {
      ss << "Including non-PyTorch memory, this process";
    } else {
      ss << "Process " << proc.pid;
    }
    ss << " has " << format_size(proc.usedGpuMemory) << " memory in use. ";
  }
  return ss.str();
#else
  return "";
#endif
}

namespace Native {

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

  // captures_underway tracks if we are diverting some
  // allocations to a specific pool.
  // Most of the time it's empty, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  std::vector<std::pair<MempoolId_t, std::function<bool(cudaStream_t)>>>
      captures_underway;

  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding cuda events
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  bool set_fraction = false;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

  std::vector<AllocatorTraceTracker> trace_trackers_;

 public:
  DeviceCachingAllocator()
      : large_blocks(/*small=*/false),
        small_blocks(/*small=*/true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CUDAAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    if (!enabled) {
      alloc_trace_next = 0;
      alloc_trace->clear();
    }
  }

  bool isHistoryEnabled() {
    return record_history;
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    size_t allocated_pool_blocks = 0;

    for (Block* b : active_blocks) {
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }

        allocated_pool_blocks += 1;
      }
    }

    return allocated_pool_blocks == expected_live_allocations.size();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    trace_trackers_.emplace_back(std::move(tracker));
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(
      c10::DeviceIndex device,
      size_t orig_size,
      cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.size() == 0)) {
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
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

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
              CUDAAllocatorConfig::garbage_collection_threshold() > 0.0)) {
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
          || (C10_LIKELY(captures_underway.size() == 0) &&
              release_cached_blocks(context) &&
              alloc_block(params, true, context, lock));
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free = 0;
      size_t device_total = 0;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
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
          std::move(context));
      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(c10::DeviceType::CUDA, device));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

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
          "CUDA out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". GPU ",
          device,
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. ",
          proc_info,
          "Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " If reserved but unallocated memory is large try setting",
          " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid"
          " fragmentation.  See documentation for Memory Management "
          " (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        std::move(params), orig_size, std::move(context), split_remainder);
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
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
          increase_stat(stats.inactive_split_bytes[stat_type], remaining->size);
          increase_stat(stats.inactive_split[stat_type], 1);
        });
      }

    } else if (already_split && !block->expandable_segment_) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        decrease_stat(stats.inactive_split_bytes[stat_type], block->size);
        decrease_stat(stats.inactive_split[stat_type], 1);
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
        block->context_when_allocated);

    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      increase_stat(stats.allocation[stat_type], 1);
      increase_stat(stats.allocated_bytes[stat_type], block->size);
      increase_stat(stats.active[stat_type], 1);
      increase_stat(stats.active_bytes[stat_type], block->size);
      increase_stat(stats.requested_bytes[stat_type], block->requested_size);
    });
    if (block->size >= CUDAAllocatorConfig::max_split_size())
      increase_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  void free(Block* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.allocation[stat_type], 1);
      decrease_stat(stats.allocated_bytes[stat_type], block->size);
    });

    record_trace(
        TraceEntry::FREE_REQUESTED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        context ? context : block->context_when_allocated);

    if (block->size >= CUDAAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_allocations, 1);

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(captures_underway.size())) {
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
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(
        !block->expandable_segment_,
        "Tensors allocated with expandable_segments:True cannot be shared between processes. Consider using expandable_segments:False in data loading workers via torch.cuda.memory._set_allocator_settings('expandable_segments:False')");
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context);
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes = 0;
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes));
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    stats.num_sync_all_streams = 0;
    stats.num_device_alloc = 0;
    stats.num_device_free = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /* Checkpoint the state of a private pool necessary to return it to its
   * current state */
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_blocks =
          get_private_pool_head_blocks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    std::unordered_map<void*, Block*> orig_ptrs_to_blocks;

    auto pool_blocks = get_private_pool_head_blocks(private_pool);

    std::vector<Block*> head_blocks;
    for (Block* block : pool_blocks) {
      if (block->prev == nullptr) {
        head_blocks.push_back(block);
      }
    }

    for (Block* block : head_blocks) {
      Block* curr = block;

      while (curr) {
        // When we free a block, its pointer should never change
        // only its adjacent blocks, so free, then look at pointer
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "Events should have synchronized when setting checkpointed block");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    for (Block* b : get_private_pool_head_blocks(private_pool)) {
      Block* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // checkpoint the state of an allocation that may have been
  // split into multiple blocks
  void setSegmentStateToCheckpoint(
      Block* block,
      SegmentState& segment,
      std::shared_ptr<GatheredContext> context,
      RestoreResult& rr) {
    Block* curr_block = block;
    Block* last_block = block;

    TORCH_INTERNAL_ASSERT(block->pool);
    BlockPool& pool = *block->pool;
    const auto segment_len = segment.blocks.size();

    // allocate all blocks in the segment
    for (size_t i = 0; i < segment_len; ++i) {
      auto& block_state = segment.blocks.at(i);
      AllocParams params(
          block_state.device,
          block_state.size,
          block_state.stream,
          &pool,
          block_state.size,
          stats);
      pool.blocks.erase(curr_block);
      params.block = curr_block;
      params.stat_types = get_stat_types_for_pool(pool);

      // splitting a block depends on `max_split_size`, which may have changed
      // between whe checkpoint was taken and now, so we make sure to recreate
      // the behavior from the checkpoint.
      bool split = (i + 1) < segment.blocks.size();

      // curr_block will become next pointer if it is split, so reassign with
      // the returned value
      curr_block = alloc_found_block(
          std::move(params), block_state.size, context, split);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->size == block_state.size);

      last_block = curr_block;
      curr_block = curr_block->next;

      TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)));
    }

    while (last_block->prev) {
      last_block = last_block->prev;
    }

    // free blocks that are not allocated in the checkpoint
    curr_block = last_block;

    for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
      auto& block_state = segment.blocks.at(i);
      TORCH_INTERNAL_ASSERT(curr_block != nullptr);

      if (block_state.allocated) {
        rr.allocations_created.push_back(curr_block);
        continue;
      }

      free(curr_block);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->allocated == block_state.allocated);
      TORCH_CHECK(curr_block->size == block_state.size);
    }
  }

  /**
   * Note [Checkpointing PrivatePoolState]
   *
   * Refer above to Note [Interaction with CUDA graph capture]. Allocations made
   * during graph capture are made from a separate private pool. During graph
   * capture allocations behave as usual. During graph replay the allocator
   * state does not change even as new tensors are created. The private pool
   * will not free its blocks to the main caching allocator until cuda graph use
   * is finished to prevent an allocation from eager clobbering the memory from
   * a live but unaccounted for tensor that was created during replay.
   *
   * `make_graphed_callables`, a series of separate callables chained in
   * successive cuda graphs, can share a memory pool because after a cuda graph
   * recording the allocations in the shared private pool exactly reflect the
   * tensors that are allocated.
   *
   * We would like to extend callable chaining to support a graphed callable
   * tree. In this scenario, we have a tree of callable chains which will be
   * captured with cuda graphs. In the diagram below, we have a tree with four
   * callables, A, B, C, and D. Suppose we have captured, and subsequently
   * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
   * would now like to record D. At this point the private pool will not reflect
   * any of the live tensors created during graph replay. Allocations made
   * during a new recording with the pool could overwrite those live tensors.
   *
   * In order to record a new graph capture after replaying prior callables in
   * the tree, we need the allocator to reflect the state of the live tensors.
   * We checkpoint the state of the private pool after each recording, and then
   * reapply it when we are starting a new recording chain. Additionally, we
   * must free the allocations for any tensors that died between the end of our
   * previous graph replaying and our new recording. All of the allocated
   * segments that existed in the checkpointed state must still exist in the
   * pool. There may also exist new allocated blocks.
   * (TODO : link note [live tensors between iterations] when it exists). For
   * every block that is currently allocated but no allocated in the snapshot,
   * we will return a pointer to their block.
   *.
   *
   *
   *  ---------------> A ---------------> B ---------------> C
   *                                      |
   *                                      |
   *                                      |
   *                                      |
   *                                      ╰ ---------------> D
   */
  RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
    // To reset the caching allocator state we will
    // - Free all the blocks currently allocated to the pool (see [live tensors
    // between iterations])
    // - Allocate all the blocks in a checkpointed segment, whether they are
    // live or not
    // - Free the blocks in a checkpointed segment which are not live
    // This could be optimized, but it nicely reuses exiting apis, and this
    // is not on the hot path.

    // following `done outside the lock because we don't know what locks the
    // recorder needs to have...`

    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::STATE);

    std::lock_guard<std::recursive_mutex> lock(mutex);

    RestoreResult rr;

    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    PrivatePool* private_pool = pool->second.get();

    freeBlocksAllocatedToPool(private_pool, rr);

    std::unordered_map<void*, Block*> ptrs_to_blocks;
    // at this point, all of the blocks should be free, so they will all be in
    // the block set
    for (Block* block : private_pool->small_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }
    for (Block* block : private_pool->large_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }

    for (auto& segment : pps.segments) {
      auto ptr = segment.blocks.at(0).ptr;
      TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr)
      auto block = ptrs_to_blocks[ptr];

      setSegmentStateToCheckpoint(block, segment, context, rr);
    }
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::unordered_map<PrivatePool*, MempoolId_t> pool_to_id;
    pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
    for (const auto& pair : graph_pools) {
      pool_to_id[pair.second.get()] = pair.first;
    }
    for (const auto& pair : graph_pools_freeable) {
      pool_to_id[pair.second] = pair.first;
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      // For expandable segments, we report one segment for each contiguous
      // mapped range of memory
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
      segment_info.is_large = (!head_block->pool->is_small);
      segment_info.is_expandable = head_block->expandable_segment_;
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;
      auto mempool_id = pool_to_id.find(head_block->pool->owner_PrivatePool);
      if (mempool_id != pool_to_id.end()) {
        segment_info.owner_private_pool_id = mempool_id->second;
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

    record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, nullptr);
    return result;
  }

  std::vector<TraceEntry> trace(
      std::function<time_t(approx_time_t)> tsc_to_us) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() + alloc_trace_next,
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() + alloc_trace_next);

    // Convert all the timestamps from tsc to epoch time in microseconds.
    for (auto& te : result) {
      te.time_.t_ = tsc_to_us(te.time_.approx_t_);
    }
    return result;
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

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
      auto divisions = CUDAAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

  // See Note [Interaction with CUDA graph capture]

  // Called by CUDAGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool. Make a new pool for
      // this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // mempool_id references an existing pool, which the current capture will
      // share. Check this pool is live (at least one other capture already
      // references it).
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
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

  // Called by CUDAGraph::reset
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
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  void addPeerAccess(c10::DeviceIndex dev_to_access) {
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

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
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

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Block* find_expandable_block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size) {
    Block key(device, stream, 0);

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
         it != pool->unmapped.end() && (*it)->stream == stream;
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
    expandable_segments_.emplace_back(new ExpandableSegment(
        device, stream, segment_size, devices_with_peer_access_));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(
      Block* to_map,
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

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block* remaining = new Block(
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

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.insert_into_blocks(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      increase_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });

    stats.num_device_alloc++;
    record_trace(
        TraceEntry::SEGMENT_MAP,
        int64_t(mapped_range.ptr),
        mapped_range.size,
        to_map->stream,
        to_map->device,
        ctx);
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = ctx;
    }

    return true;
  }

  Block* try_allocate_expandable_block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    Block* candidate = find_expandable_block(device, stream, pool, size);
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

  /** moves a block into a pool of cached free blocks */
  void free_block(
      Block* block,
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
        context ? context : block->context_when_allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);

    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segments from
      // inactive_split
      if (!block->expandable_segment_) {
        if (net_change_inactive_split_blocks > 0) {
          increase_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(net_change_inactive_split_blocks));
        } else if (net_change_inactive_split_blocks < 0) {
          decrease_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(-net_change_inactive_split_blocks));
        }
        if (net_change_inactive_split_size > 0) {
          increase_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(net_change_inactive_split_size));
        } else if (net_change_inactive_split_size < 0) {
          decrease_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(-net_change_inactive_split_size));
        }
      }
      decrease_stat(stats.active[stat_type], 1);
      decrease_stat(stats.active_bytes[stat_type], original_block_size);
      decrease_stat(stats.requested_bytes[stat_type], requested_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else { // [dest src]
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

  BlockPool& get_pool(size_t size, cudaStream_t stream) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only non-empty if some thread has begun and not yet ended
    // a capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway.size())) {
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
#endif
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small || CUDAAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CUDAAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
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

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CUDAAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      ++pool.get_free_blocks_call_count;
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;

    if ((*it)->expandable_segment_) {
      if (CUDAAllocatorConfig::expandable_segments()) {
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
    if ((p.size() < CUDAAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CUDAAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CUDAAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks(
      const std::shared_ptr<GatheredContext>& context) {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CUDAAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
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
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count() >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count(); // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block, context);
        }
      }
    }
  }

  // This function assumes that global lock has been taken whle calling into
  // this function. We do cudaMalloc sync call in this function which
  // can be expensive while holding the lock. Hence, we pass-in the lock to the
  // function to temporarily release the lock before cudaMalloc call and acquire
  // it back again after the call so that other threads dont get blocked.
  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<GatheredContext>& ctx,
      std::unique_lock<std::recursive_mutex>& lock) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else if (
        CUDAAllocatorConfig::expandable_segments() &&
        // our checkpointing logic for private pools doesn't support
        // the expandable_segments_ structure yet
        !p.pool->owner_PrivatePool) {
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.err = cudaSuccess;
      } else {
        p.err = cudaErrorMemoryAllocation;
      }
      return bool(p.block);
    } else {
      if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
        // At scope exit, acquire the lock again. This provides safety against
        // any potential exceptions in the cudaMallocMaybeCapturing function.
        auto sg = c10::make_scope_exit([&]() { lock.lock(); });
        lock.unlock();
        p.err = cudaMallocMaybeCapturing(&ptr, size);
      } else {
        p.err = cudaMallocMaybeCapturing(&ptr, size);
      }
      if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
        TORCH_CHECK(
            lock.owns_lock(), "Failed to acquire lock after cudaMalloc");
      }

      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's internal error state.
          //
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take over to throw a helpful exception. The user can choose
          // to catch the exception, free some stuff in their script, and
          // attempt the allocation again. In this case, we can also forgive and
          // clear CUDA's internal error state.
          (void)cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      increase_stat(stats.segment[stat_type], 1);
      increase_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CUDAAllocatorConfig::max_split_size())
      increase_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    stats.num_device_alloc++;
    record_trace(
        TraceEntry::SEGMENT_ALLOC,
        int64_t(p.block->ptr),
        p.block->size,
        p.stream(),
        p.device(),
        ctx);
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(
      const AllocParams& p,
      const std::shared_ptr<GatheredContext>& context) {
    if (CUDAAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    // Use constructor for search key.
    Block key(p.search_key.device, p.search_key.stream, p.search_key.size);
    key.size = (key.size < CUDAAllocatorConfig::max_split_size())
        ? CUDAAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CUDAAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur, context);
        } else {
          release_block(*cur, context);
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

  bool release_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(context);

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks, context);
    release_blocks(small_blocks, context);

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
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

  void release_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    stats.num_device_free++;
    record_trace(
        TraceEntry::SEGMENT_FREE,
        int64_t(block->ptr),
        block->size,
        block->stream,
        block->device,
        context ? context : block->context_when_segment_allocated);

    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.segment[stat_type], 1);
      decrease_stat(stats.reserved_bytes[stat_type], block->size);
    });

    if (block->size >= CUDAAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_segments, 1);
    pool->blocks.erase(block);
    delete block;
  }

  void unmap_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
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
          block->device, block->stream, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->insert_into_blocks(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block* after_free = new Block(
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

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.reserved_bytes[stat_type], unmapped.size);
    });

    stats.num_device_free++;
    record_trace(
        TraceEntry::SEGMENT_UNMAP,
        int64_t(unmapped.ptr),
        unmapped.size,
        block->stream,
        block->device,
        context ? context : block->context_when_segment_allocated);
  }
  void release_blocks(
      BlockPool& pool,
      const std::shared_ptr<GatheredContext>& context) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
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
        release_block(block, context);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block, context);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  EventPool::Event create_event_internal(c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context) {
    // Synchronize on outstanding events and then free associated blocks.
    stats.num_sync_all_streams++;

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway.size() == 0);
    insert_events_deferred_until_no_capture();

    for (auto& st : cuda_events) {
      for (auto& e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        C10_CUDA_CHECK(cudaEventSynchronize(*event));

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block) {
    c10::DeviceIndex prev_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));

      EventPool::Event event = create_event_internal(stream.device_index());
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      block->event_count++;
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events(const std::shared_ptr<GatheredContext>& context) {
    insert_events_deferred_until_no_capture();

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

        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // ignore and clear the error if not ready
          (void)cudaGetLastError();
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
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

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      cudaStream_t stream,
      c10::DeviceIndex device,
      std::shared_ptr<GatheredContext> context) {
    if (!record_history && !trace_trackers_.size())
      return;

    auto te = TraceEntry(
        action,
        device,
        addr,
        size,
        stream,
        getApproximateTime(),
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);

    // Callbacks should not include any Pytorch call
    for (const auto& cb : trace_trackers_) {
      cb(te);
    }

    if (record_history) {
      if (alloc_trace->size() < alloc_trace_max_entries_) {
        alloc_trace->emplace_back(te);
      } else {
        (*alloc_trace)[alloc_trace_next++] = te;
        if (alloc_trace_next == alloc_trace_max_entries_) {
          alloc_trace_next = 0;
        }
      }
    }
  }
};

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cudaMalloc.  This setting is useful when debugging GPU memory
// errors, since the caching allocator foils cuda-memcheck.
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(reinterpret_cast<uintptr_t>(ptr));
  }
  C10_CUDA_CHECK(cudaFree(ptr));
}

void local_raw_delete(void* ptr);

class NativeCachingAllocator : public CUDAAllocator {
 private:
  // Shard allocation region to have independent mutexes to reduce contention.
  static constexpr size_t kNumMutexShard = 67;

  // TODO: use std::hardware_destructive_interference_size once available
  struct alignas(64) AlignedMutex {
    std::mutex m;
  };

  std::array<AlignedMutex, kNumMutexShard> mutex;

  // allocated blocks by device pointer
  std::array<ska::flat_hash_map<void*, Block*>, kNumMutexShard>
      allocated_blocks;

  static size_t get_mutex_shard_id(void* ptr) {
    return twang_mix64((size_t)ptr) % kNumMutexShard;
  }

  void add_allocated_block(Block* block) {
    const auto mutex_shard_id = get_mutex_shard_id(block->ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    allocated_blocks[mutex_shard_id][block->ptr] = block;
  }

  c10::ApproximateClockToUnixTimeConverter clock_converter;

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  Block* get_allocated_block(void* ptr, bool remove = false) {
    const auto mutex_shard_id = get_mutex_shard_id(ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    auto it = allocated_blocks[mutex_shard_id].find(ptr);
    if (it == allocated_blocks[mutex_shard_id].end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks[mutex_shard_id].erase(it);
    }
    return block;
  }

  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  bool initialized() override {
    return !device_allocator.empty();
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(
      void** devPtr,
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          reinterpret_cast<uintptr_t>(block->ptr));
    }
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    C10_CUDA_CHECK(c10::cuda::SetDevice(device));
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    for (auto& allocator : device_allocator) {
      allocator->recordHistory(
          enabled, context_recorder, alloc_trace_max_entries, when);
    }
  }

  bool isHistoryEnabled() override {
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    return device_allocator[device]->isHistoryEnabled();
  }

  bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return device_allocator[device]->checkPoolLiveAllocations(
        mempool_id, expected_live_allocations);
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    for (auto& allocator : device_allocator) {
      allocator->attachOutOfMemoryObserver(std::move(observer));
    }
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
    for (auto& allocator : device_allocator) {
      allocator->attachAllocatorTraceTracker(tracker);
    }
  }

  void emptyCache() override {
    for (auto& da : device_allocator)
      da->emptyCache();
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }

  SnapshotInfo snapshot() override {
    // Set-up converter to convert timestamps from tsc to microseconds.
    auto tsc_to_ns = clock_converter.makeConverter();
    auto tsc_to_us = [=](approx_time_t t_approx) {
      return tsc_to_ns(t_approx) / 1000;
    };

    SnapshotInfo result;
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace(tsc_to_us));
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }

    auto& md = result.config_metadata;
    md.garbage_collection_threshold =
        CUDAAllocatorConfig::garbage_collection_threshold();
    md.max_split_size = CUDAAllocatorConfig::max_split_size();
    md.pinned_num_register_threads =
        CUDAAllocatorConfig::pinned_num_register_threads();
    md.expandable_segments = CUDAAllocatorConfig::expandable_segments();
    md.release_lock_on_malloc =
        CUDAAllocatorConfig::release_lock_on_cudamalloc();
    md.pinned_use_host_register =
        CUDAAllocatorConfig::pinned_use_cuda_host_register();
    md.last_allocator_settings = CUDAAllocatorConfig::last_allocator_settings();
    md.roundup_power2_divisions =
        CUDAAllocatorConfig::roundup_power2_divisions();

    return result;
  }

  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    return device_allocator[device]->getCheckpointState(id);
  }

  /**
   * @brief Checkpoint the private pool state identified in `as` to its prior
   * state
   *
   * @param device - device of the pool to manipulate
   * @param as - allocator state
   * @param stale_live_storages - storages of tensors which are currently
   * allocated but which will be not be allocated after the checkpoint is set.
   * For these storages we will remove their deleter function.
   * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
   * functions for all allocated blocks in the new checkpoint state.
   */
  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> as) override {
    std::shared_ptr<PrivatePoolState> pps =
        std::dynamic_pointer_cast<PrivatePoolState>(as);

    TORCH_CHECK(pps, "Expected PrivatePoolState");

    auto rr = device_allocator[device]->setCheckpointPoolState(*pps);

    CheckpointDelta cpd;
    for (void* ptr : rr.allocations_freed) {
      get_allocated_block(ptr, /*remove*/ true);
      cpd.ptrs_freed.push_back(ptr);
    }
    for (Block* block : rr.allocations_created) {
      add_allocated_block(block);
      cpd.dataptrs_allocd.emplace_back(
          block->ptr,
          block->ptr,
          &local_raw_delete,
          Device(DeviceType::CUDA, device));
    }

    return cpd;
  }

  DataPtr allocate(size_t size) override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* devPtr = nullptr;
    void (*deleteFunc)(void*) = &local_raw_delete;
    CUDAStream stream = cuda::getCurrentCUDAStream(device);

    if (forceUncachedAllocator()) {
      deleteFunc = &uncached_delete;

      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMalloc(&devPtr, size));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_memory_allocation(
            reinterpret_cast<uintptr_t>(devPtr));
      }
    } else {
      if (size != 0) {
        this->malloc(&devPtr, device, size, stream);
      }
    }

    if (size && TORCH_SDT_IS_ENABLED(malloc)) {
      TORCH_SDT_WITH_SEMAPHORE(malloc, devPtr, device, size, stream.id());
    }

    return {devPtr, devPtr, deleteFunc, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
    device_allocator[device]->cacheInfo(largestBlock);
  }
  void assertValidDevice(c10::DeviceIndex device) {
    const auto device_num = device_allocator.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocator[device]->getStats();
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
  }
  // CUDAGraph interactions
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) override {
    assertValidDevice(device);
    device_allocator[device]->beginAllocateToPool(
        std::move(mempool_id), std::move(filter));
  }

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    assertValidDevice(device);
    device_allocator[device]->endAllocateToPool(mempool_id);
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->releasePool(std::move(mempool_id));
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }

  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override {
    c10::cuda::CUDAGuard device_guard(dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
      // ignore and clear the error if access was already enabled
      (void)cudaGetLastError();
    } else {
      C10_CUDA_CHECK(err);
    }
    device_allocator[dev_to_access]->addPeerAccess(dev);
  }

  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override {
    if (p2p_enabled || // memcpy ok because memory is mapped in both devices
        srcDevice == dstDevice || // memcpy ok on a single device
        // memcpy ok because both dst and src must have come from cudaMalloc
        (!device_allocator[dstDevice]->hasAllocatedExpandableSegments() &&
         !device_allocator[srcDevice]->hasAllocatedExpandableSegments())) {
      return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }
    // when p2p is not enabled, only cudaMemcpyPeerAsync correctly handles
    // memory not allocated via cudaMalloc
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
  }

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  // In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
  // is called by the receiving process to map the CUDA memory from the sending
  // process into its own address space.
  //
  // CUDA IPC only allows sharing a big memory block associated with a
  // cudaIpcMemHandle_t and it can be opened only **once** per context per
  // process. There can be multiple types of storage in the same IPC mem block,
  // so we must cache the device ptr to construct typed storage as it comes.
  //
  // ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the
  // process that can be used to access the memory block in the sender process.
  // It only saves a weak_ptr of the device pointer in the map, the shared_ptr
  // will be used to reconstruct all storages in this CudaMalloc allocation. And
  // it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
  //
  std::mutex IpcMutex;
  ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex);

    auto iter = ipcMemHandle_to_devptr.find(handle);
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.lock();
      if (devptr)
        return devptr;
    }
    // This ipcMemHandle hasn't been opened, or already expired, open it to
    // enable IPC access to that mem block.
    void* dev = nullptr;
    auto ipc_handle =
        reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    // devPtr has to be deleted in same device when created.
    c10::DeviceIndex curr_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&curr_device));
    auto sp =
        std::shared_ptr<void>(dev, [handle, curr_device, this](void* ptr) {
          cuda::CUDAGuard device_guard(curr_device);
          std::lock_guard<std::mutex> deleter_lock(IpcMutex);
          C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
          ipcMemHandle_to_devptr.erase(handle);
        });
    std::weak_ptr<void> wp = sp;
    // To eliminate an additional search, we can use insert().
    // It doesn't overwrite when key already exists(ptr expired).
    // But in the deleter for sp we erased the entry,
    // this should be safe to do now.
    ipcMemHandle_to_devptr.insert(iter, {handle, wp});

    return sp;
  }
  std::string name() override {
    return "native";
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    C10_CUDA_CHECK(
        cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
  }
};

NativeCachingAllocator allocator;

void local_raw_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  allocator.free(ptr);
}

} // namespace Native
// Size pretty-printer
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

namespace CudaMallocAsync {
// If this is put in its own header file, it gets incorrectly renamed in HIPify.
CUDAAllocator* allocator();

} // namespace CudaMallocAsync

struct BackendStaticInitializer {
  // Parses env for backend at load time, duplicating some logic from
  // CUDAAllocatorConfig. CUDAAllocatorConfig double-checks it later (at
  // runtime). Defers verbose exceptions and error checks, including Cuda
  // version checks, to CUDAAllocatorConfig's runtime doublecheck. If this
  // works, maybe we should move all of CUDAAllocatorConfig here?
  CUDAAllocator* parseEnvForBackend() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "cudaMallocAsync")
              return CudaMallocAsync::allocator();
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

std::atomic<CUDAAllocator*> allocator;
BackendStaticInitializer backend_static_initializer;

} // namespace cuda::CUDACachingAllocator

} // namespace c10
