//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>

#include <c10/util/flat_hash_map.h>
#include <mach/vm_page_size.h>
#include <cstdio>
#include <mutex>
#include <set>
#include <unordered_set>

// this implementation is based on CUDACachingAllocator.
// It utilizes Metal Heaps to improve the performance with buffer allocation.
// Do not include this header. Use MPSAllocatorInterface.h instead.
// TODO: Unify the logic with CUDACachingAllocator and remove redundant code.
namespace at::mps::HeapAllocator {

static const size_t kMaxSmallAlloc = MB(1); // largest "small" allocation is 1 MiB
static const size_t kMinLargeAlloc = MB(10); // allocations between 1 and 10 MiB may use kLargeHeap
static const size_t kRoundLarge = MB(2); // round up large allocations to 2 MiB
static const size_t kSmallHeap = MB(8); // "small" allocations are packed in 8 MiB heaps
static const size_t kLargeHeap = MB(32); // "large" allocations may be packed in 32 MiB heaps
static const size_t kXLargeHeap = MB(1024); // "extra large" allocations may be packed in 1 GiB heaps
static const size_t kMaxScalarAlloc = (sizeof(int64_t)); // largest "scalar" allocation

enum class HeapTier { SMALL, LARGE, XLARGE, OVERSIZE };

inline HeapTier getHeapTier(size_t size, bool has_memory_pressure) {
  if (size <= kMaxSmallAlloc) {
    return HeapTier::SMALL;
  } else if (size < kMinLargeAlloc) {
    return HeapTier::LARGE;
  } else if (size < kXLargeHeap / 2 && !has_memory_pressure) {
    return HeapTier::XLARGE;
  }
  return HeapTier::OVERSIZE;
}

// buffer pools could be customized with a combination of usage flags
enum UsageFlags : uint32_t {
  PRIVATE = 0,
  SMALL = (1 << 0), // small heaps have sizes of kSmallHeap, and large ones kLargeHeap
  SHARED = (1 << 1), // shared (unified-memory) storage. Only ever actually used on
                      // Apple Silicon (Apple-family GPU + unified memory) — see
                      // MPSHeapAllocatorImpl::normalizeUsage(), which remaps any
                      // SHARED request to MANAGED on every other device, so this
                      // bit should never reach Metal except on Apple Silicon.
  MANAGED = (1 << 2), // managed storage mode: CPU and GPU each keep their own copy of the
                       // buffer's contents, kept in sync via didModifyRange/synchronizeResource.
                       // This is the substitute we use for SHARED on Intel/AMD (non-Apple-family)
                       // Macs, since those devices cannot back Shared-storage heaps at all.
  HAZARD = (1 << 3), // enables Automatic Hazard Tracking for the resources allocated on the pool
  SCALAR = (1 << 4), // used to import CPU scalar values to GPU and use them in MPS Stream
};

// Bitmask of usage flags whose underlying MTLBuffer exposes a valid, non-nil CPU pointer
// via `-contents`. On Apple Silicon that's SHARED; on Intel/AMD Macs we use MANAGED instead,
// since PRIVATE buffers have no CPU-visible memory at all (`-contents` is nil/unusable for
// them) and would crash any code path that dereferences `cpu_ptr`.
static constexpr uint32_t kHostAccessibleUsageMask = UsageFlags::SHARED | UsageFlags::MANAGED;

// debug verbosity flags
enum DebugVerbosity : uint32_t {
  SILENT = 0,
  PROFILING = (1 << 0), // print generic profiling data for total system memory usage
  ALLOCATIONS = (1 << 1), // print buffer allocations
  RECYCLES = (1 << 2), // print buffer recycling
  RELEASES = (1 << 3), // print buffer releases
  LARGE_ONLY = (1 << 4), // only log large buffer pool transactions
};

struct HeapBlock;

struct BufferBlock {
  id<MTLBuffer> buffer;
  void* cpu_ptr = nullptr; // stores the pointer to CPU mapping of a Shared/Managed MTLBuffer
  size_t size; // size after alignment
  size_t requested_size; // requested size (before alignment)
  // buffer shape is used for retrieving base of views in cached graphs
  std::vector<int64_t> shape;
  bool in_use = false;
  HeapBlock* heap;
  id_t buf_id;
  // counter to candidate least recently used buffers for garbage collection
  uint32_t gc_count = 0;
  uint32_t use_count = 0;
  // counter to assign unique ids to buffer blocks
  static uint64_t buffer_counter;
  // Metal events used to sync GPU/CPU operations on the shared-storage buffers
  MPSEventPtr event;
  // Stream for which this buffer was allocated.
  MPSStream* stream = nullptr;

  BufferBlock(size_t Size, size_t RequestedSize = 0, const id<MTLBuffer> Buffer = nullptr, HeapBlock* Heap = nullptr)
      : buffer(Buffer), size(Size), requested_size(RequestedSize), heap(Heap), buf_id(Buffer ? ++buffer_counter : 0) {}

  static bool Comparator(const BufferBlock* a, const BufferBlock* b) {
    return (a->size != b->size) ? a->size < b->size : (uintptr_t)a->buffer < (uintptr_t)b->buffer;
  }
  static size_t alignUp(size_t Size, size_t Alignment) {
    assert(((Alignment - 1) & Alignment) == 0);
    return ((Size + Alignment - 1) & ~(Alignment - 1));
  }
  uint32_t retainCount() const {
    return [buffer retainCount];
  }
};
typedef bool (*BufferComparison)(const BufferBlock*, const BufferBlock*);

struct BufferPool;
struct AllocParams {
  AllocParams(size_t Alloc_Size, size_t Requested_Size, BufferPool* Pool)
      : search_key(Alloc_Size), pool(Pool), requested_size(Requested_Size) {}
  size_t size() const {
    return search_key.size;
  }

  BufferBlock search_key;
  BufferPool* pool;
  BufferBlock* buffer_block = nullptr;
  size_t requested_size;
  // true if we exceed the low watermark limit. In this case
  // we apply strategies to relieve the pressure before allocation.
  bool has_memory_pressure = false;
};

struct HeapBlock {
  id<MTLHeap> heap;
  struct {
    size_t total, available;
  } size;
  BufferPool* pool;
  unsigned int n_buffers = 0;
  id_t heap_id;
  // indicates if we split this heap to sub-allocate 'several' buffers (otherwise single buffer)
  bool is_split;
  // counter to assign unique ids to heap blocks
  static uint64_t heap_counter;

  HeapBlock(size_t Size, const id<MTLHeap> Heap = nullptr, BufferPool* Pool = nullptr)
      : heap(Heap),
        size({.total = Size, .available = Size}),
        pool(Pool),
        heap_id(Heap ? ++heap_counter : 0),
        is_split(true) {}

  static MTLResourceOptions getOptions(uint32_t usage) {
    // TODO: check the caching performance of write-combined mode
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;

    if (usage & UsageFlags::MANAGED)
      options |= MTLResourceStorageModeManaged;
    else if (usage & UsageFlags::SHARED)
      options |= MTLResourceStorageModeShared;
    else
      options |= MTLResourceStorageModePrivate;

    options |=
        (usage & UsageFlags::HAZARD) ? MTLResourceHazardTrackingModeTracked : MTLResourceHazardTrackingModeUntracked;

    return options;
  }

  static HeapBlock* createHeapBlock(AllocParams& params, id<MTLDevice> device, uint32_t usage) {
    HeapBlock* heapBlock = nullptr;
    if (!device) {
      return nullptr; // defensive: no Metal device to allocate from
    }
    bool is_split = true;
    const size_t size = params.size();
    MTLHeapDescriptor* d = [MTLHeapDescriptor new];
    if (d) {
      switch (getHeapTier(size, params.has_memory_pressure)) {
        case HeapTier::SMALL:
          d.size = kSmallHeap;
          break;
        case HeapTier::LARGE:
          d.size = kLargeHeap;
          break;
        case HeapTier::XLARGE:
          d.size = kXLargeHeap;
          break;
        case HeapTier::OVERSIZE:
          d.size = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
          is_split = false;
          break;
      }
      // Mirror getOptions()'s priority: MANAGED > SHARED > PRIVATE. `usage` here
      // is expected to have already passed through
      // MPSHeapAllocatorImpl::normalizeUsage()/hostAccessibleUsage(), so on a
      // non-Apple-family device this will never end up MTLStorageModeShared.
      d.storageMode = (usage & UsageFlags::MANAGED) ? MTLStorageModeManaged
                      : (usage & UsageFlags::SHARED) ? MTLStorageModeShared
                                                      : MTLStorageModePrivate;
      d.cpuCacheMode = MTLCPUCacheModeDefaultCache;
      // this automatically handles Metal buffer access synchronizations at the
      // cost of slightly lower performance.
      d.hazardTrackingMode =
          (usage & UsageFlags::HAZARD) ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
      // NOTE: do NOT also set d.resourceOptions here. MTLHeapDescriptor.resourceOptions
      // is a legacy/deprecated combined property kept only for backward compatibility;
      // setting it on top of the explicit storageMode/cpuCacheMode/hazardTrackingMode
      // above makes the driver re-decode those bits, and on Managed-storage heaps (the
      // path actually exercised on Intel/AMD Macs) that re-decoding does not reliably
      // round-trip back to a valid MTLStorageMode on every driver — surfacing as
      // "Invalid storageMode (Invalid)" heap-descriptor validation failures. This was
      // previously masked because Intel Macs always hit the (also wrong) Shared-storage
      // validation failure first. getOptions(usage) is still used correctly elsewhere,
      // for the actual MTLResourceOptions argument to newBufferWithLength:options:.
      d.type = MTLHeapTypeAutomatic;
      id<MTLHeap> heap = [device newHeapWithDescriptor:d];
      if (heap) {
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
        const size_t heap_size = heapAvailableSize(heap);
        heapBlock = new HeapBlock(heap_size, heap, params.pool);
        if (heapBlock) {
          heapBlock->is_split = is_split;
        }
      }
      [d release];
    }
    // heapBlock stays nullptr if `d` or `heap` creation failed (e.g. out of GPU
    // memory) — callers must treat that as an ordinary allocation failure, not
    // assert/crash. This is especially relevant on Intel Macs, where discrete
    // GPU VRAM is a small fixed pool and heap creation can fail far more often
    // than on Apple Silicon's large unified memory pool.
    return heapBlock;
  }
  static bool Comparator(const HeapBlock* a, const HeapBlock* b) {
    return (a->size.available != b->size.available) ? a->size.available < b->size.available
                                                    : (uintptr_t)a->heap < (uintptr_t)b->heap;
  }
  static NSUInteger heapAvailableSize(id<MTLHeap> heap, size_t Alignment = vm_page_size) {
    return [heap maxAvailableSizeWithAlignment:Alignment];
  }
  NSUInteger Size() {
    return [heap size];
  }
  id<MTLBuffer> newMTLBuffer(size_t length, uint32_t usage) {
    id<MTLBuffer> buf = [heap newBufferWithLength:length options:getOptions(usage)];
    if (buf) {
      updateAvailableSize();
      n_buffers++;
    }
    return buf;
  }
  // returns the retainCount before releasing the buffer
  uint32_t releaseMTLBuffer(id<MTLBuffer>& buffer) {
    const uint32_t retainCount = [buffer retainCount];
    [buffer release];
    buffer = nil;
    updateAvailableSize();
    n_buffers--;
    return retainCount;
  }
  // returns the retainCount before releasing the heap
  uint32_t releaseMTLHeap() {
    const uint32_t retainCount = [heap retainCount];
    TORCH_INTERNAL_ASSERT(!n_buffers); // assert if heap isn't empty
    [heap setPurgeableState:MTLPurgeableStateEmpty];
    [heap release];
    heap = nil;
    size.available = 0;
    return retainCount;
  }
  uint32_t retainCount() const {
    return [heap retainCount];
  }
  void updateAvailableSize() {
    size.available = heapAvailableSize(heap);
  }
};
typedef bool (*HeapComparison)(const HeapBlock*, const HeapBlock*);

struct BufferPool {
  enum class Kind {
    SHARED_SMALL,
    SHARED_LARGE,
    SCALAR,
  };

  BufferPool(const id<MTLDevice> Device, uint32_t Usage)
      : device(Device), usage(Usage), heaps(HeapBlock::Comparator), available_buffers(BufferBlock::Comparator) {}

  const id<MTLDevice> device;
  // usage flags to customize the pool for various purposes (see UsageFlags enum).
  // Note: the Kind names (SHARED_SMALL/SHARED_LARGE) are historical pool *roles*,
  // not a guarantee of literal Shared storage — on Intel/AMD devices these pools
  // are actually backed by Managed storage (see
  // MPSHeapAllocatorImpl::hostAccessibleUsage()/normalizeUsage()). This `usage`
  // field is always the already-resolved, device-correct value.
  const uint32_t usage;
  // total number of buffers in the pool
  uint32_t n_buffers = 0;
  // total allocations size on this pool
  size_t allocated_size = 0;
  // total memory available in the pool
  size_t available_size = 0;
  // list of heaps ordered by their "available" (not total) memory size
  std::set<HeapBlock*, HeapComparison> heaps;
  // list of only "available" buffers in the pool (i.e., buffers not in-use)
  std::set<BufferBlock*, BufferComparison> available_buffers;
  // List of available buffers partitioned by which stream allocated them.
  // Buffers can only be reused by the same stream. Allowing buffers to be used
  // by a different stream would require an expensive cross-stream
  // synchronization.
  ska::flat_hash_map<MPSStream*, std::set<BufferBlock*, BufferComparison>> available_buffers_by_stream;
  // list of buffers that are in a state of "limbo" where they've already been freed
  // from PyTorch-side, but were not returned to pool due to still being
  // in-use by command buffers with retainCount > 1. In this state, the buffer is
  // neither ready to be recycled, nor could be returned to pool as available.
  // These buffers will be returned to pool once the command buffer's
  // completionHandler callbacks are called.
  std::unordered_set<BufferBlock*> buffers_pending_free;
  // list of heaps pending size update
  std::unordered_set<HeapBlock*> heaps_pending_update;
};

class MPSHeapAllocatorImpl {
 public:
  explicit MPSHeapAllocatorImpl()
      : m_device(at::mps::MPSDevice::getInstance()->device()),
        // IMPORTANT: [device hasUnifiedMemory] is YES on Apple Silicon, but it
        // is ALSO YES on many Intel Macs that only have an Intel integrated
        // GPU (Intel iGPUs physically share system RAM with the CPU too — that
        // property answers "does this GPU share RAM with the CPU", not "is
        // this an Apple-family GPU"). Relying on hasUnifiedMemory alone caused
        // Shared-storage heaps to be requested on such Intel Macs, which Metal
        // rejects with "Shared storage mode disallowed" (Shared/Managed heaps
        // are only valid on Apple-family GPUs).
        //
        // We therefore require BOTH hasUnifiedMemory AND Apple-GPU-family
        // support. This is computed once here and is the single source of
        // truth consulted by hostAccessibleUsage()/normalizeUsage() below, so
        // every other Intel/AMD (non-Apple-family) device — unified-memory-
        // reporting or not — always falls back to Managed (or Private), and
        // only genuine Apple Silicon ever gets Shared storage.
        m_has_unified_memory(m_device != nil &&
                              [m_device hasUnifiedMemory] &&
                              [m_device supportsFamily:MTLGPUFamilyApple1]),
        m_max_buffer_size(m_device != nil ? [m_device maxBufferLength] : 0),
        m_stream(getDefaultMPSStream()),
        m_event_pool(getMPSEventPool()) {
    init_allocator();
  }
  ~MPSHeapAllocatorImpl() {
    emptyCache();
    if (m_scalar_staging_buffer) {
      [m_scalar_staging_buffer release];
      m_scalar_staging_buffer = nil;
    }
  }
  // interface exposed to at::Allocator
  id<MTLBuffer> malloc(size_t size, uint32_t usage);
  // frees a buffer and returns it into buffer pool
  void free(void* ptr);
  // releases all the cached buffers and their associated heaps
  void emptyCache();
  // free inactive buffers that are pending to be freed
  void freeInactiveBuffers();
  // returns true if buffer was allocated from a CPU-accessible pool
  // (Shared on Apple Silicon, Managed on Intel/AMD)
  bool isSharedBuffer(const void* ptr);
  // get the requested unaligned size of an MTLBuffer
  ssize_t getUnalignedBufferSize(const void* ptr);
  // set the shape of a base tensor from a view tensor
  void setBufferShape(const void* ptr, const IntArrayRef& shape);
  // retrieve the shape of a base tensor from a view tensor
  IntArrayRef getBufferShape(const void* ptr);
  // get the unique ID of the buffer
  id_t getBufferId(const void* ptr);
  // allocate a buffer from a specialized pool to import CPU scalars into GPU
  id<MTLBuffer> allocScalarBufferWithValue(void* value, size_t size);
  // returns a CPU-mapping of the input buffer and its retainCount,
  // if only it has Shared/Managed storage-mode and allocated on MPSAllocator
  std::pair<const void*, uint32_t> getSharedBufferPtr(const void* buffer);
  // returns a CPU-device c10::Storage aliasing the host-visible contents of
  // the MTLBuffer backing `mps_storage`. The returned storage keeps the
  // source MPS storage alive for its lifetime. Raises if `mps_storage` is
  // not MPS-allocated or not host-accessible (Shared/Managed) storage.
  c10::Storage getHostAliasStorage(const c10::Storage& mps_storage);
  // records events for a list of MTLBuffers (list is used to lock the mutex once)
  // returns true if records any event (given if passed buffers exist and are host-accessible)
  bool recordEvents(c10::ArrayRef<const void*> buffers);
  // waits for the event to signal the completion of GPU execution
  // on the passed host-accessible buffers (list is used to lock the mutex once)
  // returns true if actually waited on any event
  bool waitForEvents(c10::ArrayRef<const void*> buffers);
  // this indicates how far (in Megabytes) the current total allocations are from the
  // low watermark limit which is used to detect if we're under memory pressure
  // This returns zero if we've reached the low watermark limit
  ssize_t getLowWatermarkValue();
  // (see m_low_watermark_ratio for description)
  void setLowWatermarkRatio(double ratio);
  // (see m_high_watermark_ratio for description)
  void setHighWatermarkRatio(double ratio);
  // (see m_low_watermark_limit for description)
  size_t getLowWatermarkLimit() const {
    return m_low_watermark_limit;
  }
  // (see m_max_total_allowed_size for description)
  size_t getHighWatermarkLimit() const {
    return m_max_total_allowed_size;
  }
  // (see m_total_allocated_memory for description)
  size_t getTotalAllocatedMemory() const {
    return m_total_allocated_memory.current;
  }
  // (see m_current_allocated_memory for description)
  size_t getCurrentAllocatedMemory() const {
    return m_current_allocated_memory.current;
  }
  // snapshot of memory stats for the generic torch.accelerator memory APIs
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats();
  void resetAccumulatedStats();
  void resetPeakStats();
  // total GPU memory allocated in the process by Metal driver; including
  // implicit allocations from MPS/MPSGraph frameworks and MPSHeapAllocatorImpl.
  size_t getDriverAllocatedMemory() const {
    return current_allocated_size();
  }
  // recommended Max memory for Metal
  size_t getRecommendedMaxMemory() const {
    return max_device_size();
  }
  // (see enum DebugVerbosity for description)
  uint32_t getDebugVerbosity() const {
    return m_debug_verbosity;
  }
  // returns the device that we allocate from
  inline id<MTLDevice> Device() const {
    return m_device;
  }
  // true on Apple Silicon (unified memory + Apple GPU family); false on
  // Intel/AMD Macs (discrete or non-Apple-family GPUs), REGARDLESS of what
  // [device hasUnifiedMemory] reports on its own. Buffer pools use Shared
  // storage in the former case and fall back to Managed storage in the
  // latter (see hostAccessibleUsage()/normalizeUsage()).
  inline bool hasUnifiedMemory() const {
    return m_has_unified_memory;
  }

  inline std::string format_size(uint64_t size) const;

 private:
  // (see m_high_watermark_ratio for description) — Apple Silicon (unified memory) default
  constexpr static double default_high_watermark_ratio = 1.7;
  // Intel/AMD (discrete GPU) default: VRAM is a small, fixed pool, unlike the large unified
  // pool on Apple Silicon, so we don't allow over-committing past the recommended size.
  constexpr static double default_high_watermark_ratio_intel = 1.0;
  // we set the allowed upper bound to twice the size of recommendedMaxWorkingSetSize.
  constexpr static double default_high_watermark_upper_bound = 2.0;
  // (see m_low_watermark_ratio for description) — Apple Silicon (unified memory) default
  // on unified memory, we could allocate beyond the recommendedMaxWorkingSetSize
  constexpr static double default_low_watermark_ratio = 1.4;
  // Intel/AMD (discrete GPU) default: stay conservatively under the recommended size, since
  // there's no unified pool to fall back on.
  constexpr static double default_low_watermark_ratio_intel = 0.95;

  const id<MTLDevice> m_device;
  // true ONLY on genuine Apple Silicon (Apple-family GPU AND unified memory).
  // Computed once at construction time and used as the single source of truth
  // for every Shared-vs-Managed decision in this allocator. See the
  // constructor above for why this is NOT simply `[m_device hasUnifiedMemory]`.
  const bool m_has_unified_memory;
  std::recursive_mutex m_mutex;
  // allocated buffers by device pointer
  ska::flat_hash_map<const void*, BufferBlock*> m_allocated_buffers;
  // using a container for pools to simplify iterating them
  ska::flat_hash_map<BufferPool::Kind, std::unique_ptr<BufferPool>> m_pools;
  // total memory allocated by HeapAllocator (including blocks in pools);
  // tracked as a Stat to expose current/peak/accumulated reserved bytes
  c10::CachingAllocator::Stat m_total_allocated_memory;
  // currently active memory allocations in use (i.e., blocks not in pools);
  // tracked as a Stat to expose current/peak/accumulated allocated bytes
  c10::CachingAllocator::Stat m_current_allocated_memory;
  // max buffer size allowed by Metal
  size_t m_max_buffer_size = 0;
  // maximum total size allowed to be allocated
  size_t m_max_total_allowed_size = 0;
  // high watermark ratio is a hard limit for the total allowed allocations
  // 0. : disables high watermark limit (may cause system failure if system-wide OOM occurs)
  // 1. : recommended maximum allocation size (i.e., device.recommendedMaxWorkingSetSize)
  // >1.: allows limits beyond the device.recommendedMaxWorkingSetSize
  // e.g., value 0.95 means we allocate up to 95% of recommended maximum
  // allocation size; beyond that, the allocations would fail with OOM error.
  double m_high_watermark_ratio;
  // low watermark ratio is a soft limit to attempt limiting memory allocations up to the lower watermark
  // level by garbage collection or committing command buffers more frequently (a.k.a, adaptive commit).
  // Value between 0 to m_high_watermark_ratio (setting 0.0 disables adaptive commit and garbage collection)
  // e.g., value 0.9 means we 'attempt' to limit allocations up to 90% of recommended maximum
  // allocation size.
  double m_low_watermark_ratio;
  // low watermark size limit (in Bytes) at the time we initialize the allocator
  size_t m_low_watermark_limit;
  // use "PYTORCH_DEBUG_MPS_ALLOCATOR" env-var to set debug verbosity
  uint32_t m_debug_verbosity;
  // default MPS stream
  MPSStream* m_stream;
  // we hold a reference to MPSEventPool so it could get destroyed after MPSAllocator
  std::shared_ptr<MPSEventPool> m_event_pool;
  // Lazily-created, reused, standalone (non-heap) Managed staging buffer used
  // by uploadScalarViaBlit() to get a CPU-supplied scalar value into a
  // Private-storage heap buffer on non-Apple-Silicon devices. nil/unused on
  // Apple Silicon, where the scalar pool is Shared and memcpy works directly.
  // Released in the destructor.
  id<MTLBuffer> m_scalar_staging_buffer = nil;

  void init_allocator();
  void init_buffer_pools();
  HeapBlock* get_free_heap(AllocParams& params);
  bool get_free_buffer(AllocParams& params);
  BufferBlock* get_allocated_buffer_block(const void* ptr);
  BufferBlock* alloc_buffer_block(size_t size, uint32_t usage);
  bool alloc_buffer(AllocParams& params);
  void free_buffer(BufferBlock* buffer_block);
  // returns true if the container heap is also released
  bool release_buffer(BufferBlock* buffer_block, bool remove_empty_heap = true);
  // adds/removes a buffer in both `pool.available_buffers` and
  // `pool.available_buffers_by_stream[buffer_block->stream]`, so the two stay
  // in sync. `insert_available_buffer` returns whether the buffer was newly
  // inserted.
  bool insert_available_buffer(BufferPool& pool, BufferBlock* buffer_block);
  void erase_available_buffer(BufferPool& pool, BufferBlock* buffer_block);
  void release_buffers(BufferPool& pool);
  bool release_available_cached_buffers(AllocParams& params);
  bool release_cached_buffers();
  // waits for buffers parked in-flight in the pool's pending-free list to finish
  // on the GPU and returns them to the pool; returns true if any were reclaimed
  bool wait_for_pending_free_buffers(BufferPool& pool);
  // free unused cached blocks to reclaim GPU memory if memory pressure is high
  void garbage_collect_cached_buffers(AllocParams& params);
  // returns the suitable buffer pool type for the usage or
  // requested/allocated sizes
  BufferPool& get_pool(size_t requested_size, size_t aligned_size, uint32_t usage);
  // returns the aligned allocation size that is optimized
  // for the buffers to get reused frequently
  size_t get_allocation_size(size_t size, uint32_t usage) const;
  // returns the usage-flag combination for the device's general-purpose
  // (heap-backed) buffer pools. IMPORTANT: per Apple's own MTLHeap
  // documentation, MTLHeapDescriptor.storageMode disallows
  // MTLStorageModeManaged (and Memoryless) — Heaps may ONLY be Shared or
  // Private, on every platform, Apple Silicon included. So the only
  // CPU-accessible heap storage mode that exists at all is Shared, and it is
  // only valid on Apple-family GPUs with unified memory. On every other
  // device (Intel/AMD) the general pools fall back to plain PRIVATE
  // (GPU-only — no `cpu_ptr`, no pinned-memory aliasing for ordinary
  // tensors, which honestly reflects what Metal actually allows there). The
  // one call site that still needs to get a CPU-supplied value into GPU
  // memory on such devices (allocScalarBufferWithValue) does NOT rely on a
  // heap-backed Managed buffer (impossible) — it uses a small standalone,
  // non-heap MTLBuffer with Managed storage instead (Managed storage *is*
  // valid for an individually-allocated buffer; the restriction is specific
  // to Heaps) plus a blit copy. See uploadScalarViaBlit().
  inline uint32_t hostAccessibleUsage() const {
    return m_has_unified_memory ? UsageFlags::SHARED : UsageFlags::PRIVATE;
  }
  // Single funnel point: resolves any caller-requested SHARED or MANAGED bit
  // to whatever this device's HEAPS can actually back (see
  // hostAccessibleUsage()), regardless of which literal bit the caller
  // happened to pass in. This is what guarantees Intel/AMD Macs never end up
  // requesting a Shared- or Managed-storage heap — even though the
  // registered c10 MPSAllocator (the "default" allocator wired into
  // c10::SetAllocator) is constructed with a compile-time-literal
  // UsageFlags::SHARED, that literal gets remapped here to PRIVATE on
  // non-Apple-Silicon devices before it ever reaches Metal. PRIVATE usage
  // (neither SHARED nor MANAGED set) passes through unchanged.
  inline uint32_t normalizeUsage(uint32_t usage) const {
    if (usage & (UsageFlags::SHARED | UsageFlags::MANAGED)) {
      usage &= ~(UsageFlags::SHARED | UsageFlags::MANAGED);
      usage |= hostAccessibleUsage();
    }
    return usage;
  }
  // Uploads `size` bytes from a CPU-side `value` into `dst`, a Private,
  // heap-backed MTLBuffer that itself has no CPU pointer. Used only on
  // non-Apple-Silicon devices, where the scalar pool (like every other pool)
  // is Private — see hostAccessibleUsage(). Copies the value into a small,
  // lazily-created, reusable *standalone* (non-heap) Managed staging buffer
  // — legal, since the Managed-storage restriction applies to Heaps, not to
  // individually-allocated buffers — then blits from staging into `dst`.
  void uploadScalarViaBlit(id<MTLBuffer> dst, const void* value, size_t size);
  // On non-Apple-Silicon, brings the CPU-visible copy of a Managed buffer's
  // contents up to date with any pending GPU writes. Kept for any
  // standalone (non-heap) Managed buffer that may need it; a no-op on Apple
  // Silicon, where Shared storage is always coherent. NOTE: as of the
  // PRIVATE-heap fix above, no *heap-backed* buffer is ever Managed on any
  // platform, so this currently only matters for standalone Managed buffers
  // such as the scalar staging buffer.
  void synchronizeManagedBuffer(BufferBlock* buffer_block);
  // maximum size of device memory available for allocation in current process
  // Note: the recommendedMaxWorkingSetSize is typically 75% of the total system memory.
  size_t max_device_size() const {
    return m_device != nil ? [m_device recommendedMaxWorkingSetSize] : 0;
  }
  // there are implicit allocations from MPS backend, so we need to query the 'device' for
  // total allocated size instead of manually tracking in MPSAllocator
  size_t current_allocated_size() const {
    return m_device != nil ? [m_device currentAllocatedSize] : 0;
  }

  bool trigger_memory_callbacks(BufferBlock* buffer_block, IMpsAllocatorCallback::EventType event) const {
    for (const auto& name : MPSAllocatorCallbacksRegistry()->Keys()) {
      MPSAllocatorCallbacksRegistry()->Create(name)->executeMPSAllocatorCallback(
          buffer_block ? buffer_block->buffer : nullptr, event);
    }
    return true;
  }
};

} // namespace at::mps::HeapAllocator