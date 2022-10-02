//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>
#include <cstdio>
#include <mutex>
#include <set>
#include <atomic>
#include <mach/vm_page_size.h>
#include <c10/util/flat_hash_map.h>

// this implementation is based on CUDACachingAllocator.
// It utilizes Metal Heaps to improve the performance with buffer allocation.
// TODO: Unify the logic with CUDACachingAllocator and remove redundant code.
namespace at {
namespace mps {

class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED, // buffer got allocated to be used immediately
    RECYCLED,  // buffer pulled from free list to be reused
    FREED,     // buffer put to free list for future recycling
    RELEASED,  // buffer memory released
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory is freed.
C10_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__);

namespace HeapAllocator {

#define MB(x) round_page(x * 1048576UL)

static const size_t kMaxSmallAlloc = MB(1);  // largest "small" allocation is 1 MiB
static const size_t kMinLargeAlloc = MB(10); // allocations between 1 and 10 MiB may use kLargeHeap
static const size_t kSmallHeap     = MB(8);  // "small" allocations are packed in 8 MiB heaps
static const size_t kLargeHeap     = MB(32); // "large" allocations may be packed in 32 MiB heaps
static const size_t kRoundLarge    = MB(2);  // round up large allocations to 2 MiB

// buffer pools could be customized with a combination of usage flags
enum UsageFlags : uint32_t {
  PRIVATE = 0,
  SMALL   = (1 << 0), // small heaps have sizes of kSmallHeap, and large ones kLargeHeap
  SHARED  = (1 << 1), // shared pools allocated on devices with unified memory; otherwise, private between host/device
  MANAGED = (1 << 2), // managed storage mode
  HAZARD  = (1 << 3), // enables Automatic Hazard Tracking for the resources allocated on the pool
  SCALAR  = (1 << 4), // used to import CPU scalar values to GPU and use them in MPS Stream
};
// debug verbosity flags
enum DebugVerbosity : uint32_t {
  SILENT      = 0,
  PROFILING   = (1 << 0), // print generic profiling data for total system memory usage
  ALLOCATIONS = (1 << 1), // print buffer allocations
  RECYCLES    = (1 << 2), // print buffer recycling
  RELEASES    = (1 << 3), // print buffer releases
};

struct HeapBlock;

struct BufferBlock
{
  id<MTLBuffer> buffer;
  size_t size; // size after alignment
  size_t requested_size; // requested size (before alignment)
  // buffer shape is used for retrieving base of views in cached graphs
  std::vector<int64_t> shape;
  bool in_use;
  HeapBlock* heap;
  id_t buf_id;

  BufferBlock(size_t Size, size_t RequestedSize = 0, const id<MTLBuffer> Buffer = nullptr,
              HeapBlock* Heap = nullptr, id_t BufID = 0) :
              buffer(Buffer), size(Size), requested_size(RequestedSize),
              in_use(false), heap(Heap), buf_id(BufID) { }

  static bool Comparator(const BufferBlock* a, const BufferBlock* b) {
    return (a->size != b->size) ? a->size < b->size : (uintptr_t)a->buffer < (uintptr_t)b->buffer;
  }
  static size_t alignUp(size_t Size, size_t Alignment) {
    assert(((Alignment - 1) & Alignment) == 0);
    return ((Size + Alignment - 1) & ~(Alignment - 1));
  }
};
typedef bool (*BufferComparison)(const BufferBlock*, const BufferBlock*);

struct BufferPool;

struct HeapBlock
{
  id<MTLHeap> heap;
  struct { size_t total, available; } size;
  BufferPool* pool;
  unsigned int n_buffers;

  HeapBlock(size_t Size, const id<MTLHeap> Heap = nullptr, BufferPool *Pool = nullptr) :
            heap(Heap), size({.total = Size, .available = Size}), pool(Pool), n_buffers(0) { }

  static MTLResourceOptions getOptions(uint32_t usage) {
    // TODO: check the caching performance of write-combined mode
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;

    if (usage & UsageFlags::MANAGED)
      options |= MTLResourceStorageModeManaged;
    else if (usage & UsageFlags::SHARED)
      options |= MTLResourceStorageModeShared;
    else
      options |= MTLResourceStorageModePrivate;

      options |= (usage & UsageFlags::HAZARD) ? MTLResourceHazardTrackingModeTracked : MTLResourceHazardTrackingModeUntracked;

    return options;
  }

  static id<MTLHeap> createMTLHeap(id<MTLDevice> device, size_t size, uint32_t usage) {
    id<MTLHeap> heap = nil;
    MTLHeapDescriptor *d = [MTLHeapDescriptor new];
    if (d) {
      if (size <= kMaxSmallAlloc) {
        d.size = kSmallHeap;
      } else if (size < kMinLargeAlloc) {
        d.size = kLargeHeap;
      } else {
        d.size = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
      }
      d.storageMode = (usage & UsageFlags::SHARED) ? MTLStorageModeShared : MTLStorageModePrivate;
      d.cpuCacheMode = MTLCPUCacheModeDefaultCache;
      // this automatically handles Metal buffer access synchronizations at the
      // cost of slightly lower performance.
      d.hazardTrackingMode = (usage & UsageFlags::HAZARD) ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
      d.resourceOptions = getOptions(usage);
      d.type = MTLHeapTypeAutomatic;
      heap = [device newHeapWithDescriptor: d];
      if (heap) {
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
      }
      [d release];
    }
    return heap;
  }
  static bool Comparator(const HeapBlock* a, const HeapBlock* b) {
    return a->size.available < b->size.available;
  }
  static NSUInteger heapAvailableSize(id<MTLHeap> heap, size_t Alignment = vm_page_size) {
      return [heap maxAvailableSizeWithAlignment:Alignment];
  }
  id<MTLBuffer> newMTLBuffer(size_t length, uint32_t usage) {
    id<MTLBuffer> buf = [heap newBufferWithLength:length options:getOptions(usage)];
    if (buf) {
      size.available = heapAvailableSize(heap);
      n_buffers++;
    }
    return buf;
  }
  void releaseMTLBuffer(id<MTLBuffer> buffer) {
    [buffer release];
    size.available = heapAvailableSize(heap);
    n_buffers--;
  }
  void releaseMTLHeap() {
    TORCH_INTERNAL_ASSERT(!n_buffers); // assert if heap isn't empty
    [heap release];
    size.available = 0;
  }
};
typedef bool (*HeapComparison)(const HeapBlock*, const HeapBlock*);

struct BufferPool
{
  BufferPool(const id<MTLDevice> Device, uint32_t Usage) :
             device(Device), usage(Usage), n_buffers(0), allocated_size(0),
             heaps(HeapBlock::Comparator), buffers(BufferBlock::Comparator) { }

  const id<MTLDevice> device;
  // usage flags to customize the pool for various purposes (see UsageFlags enum)
  const uint32_t usage;
  // total number of buffers in the pool
  uint32_t n_buffers;
  // total allocations size on this pool
  size_t allocated_size;
  // list of heaps ordered by their "available" (not total) memory size
  std::set<HeapBlock*, HeapComparison> heaps;
  // list of only "available" buffers in the pool (i.e., buffers not in-use)
  std::set<BufferBlock*, BufferComparison> buffers;
};

struct AllocParams
{
  AllocParams(size_t Alloc_Size, size_t Requested_Size, BufferPool* Pool) :
              search_key(Alloc_Size), pool(Pool),
              buffer_block(nullptr), requested_size(Requested_Size) { }
  size_t size() const { return search_key.size; }

  BufferBlock search_key;
  BufferPool* pool;
  BufferBlock* buffer_block;
  size_t requested_size;
};

class MPSHeapAllocatorImpl
{
public:
  // (see m_high_watermark_ratio for description)
  constexpr static double default_high_watermark_ratio = 0.0;

  explicit MPSHeapAllocatorImpl() :
    m_device(at::mps::MPSDevice::getInstance()->device()),
    m_large_pool_shared (m_device, UsageFlags::SHARED  | UsageFlags::HAZARD),
    m_large_pool_private(m_device, UsageFlags::PRIVATE | UsageFlags::HAZARD),
    m_small_pool_shared (m_device, UsageFlags::SMALL   | UsageFlags::SHARED  | UsageFlags::HAZARD),
    m_small_pool_private(m_device, UsageFlags::SMALL   | UsageFlags::PRIVATE | UsageFlags::HAZARD),
    // no Hazard Tracking required for the Scalar pool (synchronized manually)
    m_scalar_pool(m_device, UsageFlags::SMALL | UsageFlags::SHARED | UsageFlags::SCALAR),
    m_total_allocated_memory(0), m_max_buffer_size([m_device maxBufferLength]),
    m_stream(getDefaultMPSStream())
  {
    init_allocator_params();
  }

  // interface exposed to at::Allocator
  id<MTLBuffer> malloc(size_t size, uint32_t usage);
  void free(void* ptr);
  void emptyCache();
  // interface exposed to internal MPS operations
  bool isSharedBuffer(void* ptr);
  ssize_t getRequestedBufferSize(void* ptr);
  void setBufferShape(void* ptr, const IntArrayRef& shape);
  IntArrayRef getBufferShape(void* ptr);
  id<MTLBuffer> allocScalarBufferWithValue(void* value, size_t size);

  bool getDebugVerbosity() const { return m_debug_verbosity; }
  size_t getMaxTotalAllowedSize() const { return m_max_total_allowed_size; }
  inline id<MTLDevice> Device() const { return m_device; }

private:
  const id<MTLDevice> m_device;
  std::mutex m_mutex;
  // allocated buffers by device pointer
  ska::flat_hash_map<void*, BufferBlock*> m_allocated_buffers;
  // unallocated cached buffers larger than 1 MB
  BufferPool m_large_pool_shared, m_large_pool_private;
  // unallocated cached buffers 1 MB or smaller
  BufferPool m_small_pool_shared, m_small_pool_private;
  // small cached buffers to import scalar values into MPS stream
  BufferPool m_scalar_pool;
  // total memory allocated by HeapAllocator
  size_t m_total_allocated_memory;
  // max buffer size allowed by Metal
  size_t m_max_buffer_size;
  // maximum total size allowed to be allocated
  size_t m_max_total_allowed_size;
  // upper limit ratio to limit the total allocations (between 0 and 1)
  // 0 means unlimited (would spill to disk or system failure if OOM)
  // 1 is maximum allowed by device.recommendedMaxWorkingSetSize
  // (e.g., value 0.95 means we allocate up to 95% of total memory; beyond that allocations fail)
  double m_high_watermark_ratio;
  // use "PYTORCH_DEBUG_MPS_ALLOCATOR" env-var to set debug verbosity
  uint32_t m_debug_verbosity;
  // default MPS stream
  MPSStream* m_stream;

  void init_allocator_params();
  HeapBlock* get_free_heap(AllocParams& p);
  bool get_free_buffer(AllocParams& p);
  BufferBlock* get_allocated_buffer_block(void* ptr);
  BufferBlock* alloc_buffer_block(size_t size, uint32_t usage);
  bool alloc_buffer(AllocParams& p);
  void free_buffer(BufferBlock* buffer_block);
  void release_buffer(BufferBlock* buffer_block, bool remove_empty_heap = true);
  void release_buffers(BufferPool& pool);
  bool release_available_cached_buffers(AllocParams& p);
  bool release_cached_buffers();

  BufferPool& get_pool(size_t Size, uint32_t usage) {
    if (usage & UsageFlags::SCALAR)
      return m_scalar_pool;
    return Size <= kMaxSmallAlloc ? ((usage & UsageFlags::SHARED) ? m_small_pool_shared : m_small_pool_private) :
                                    ((usage & UsageFlags::SHARED) ? m_large_pool_shared : m_large_pool_private);
  }

  size_t get_allocation_size(size_t Length, uint32_t usage) const  {
    MTLSizeAndAlign sizeAlign = [m_device heapBufferSizeAndAlignWithLength:Length
                                                                   options:HeapBlock::getOptions(usage)];
    return BufferBlock::alignUp(sizeAlign.size, sizeAlign.align);
  }
  // maximum size of device memory available for allocation in current process
  size_t max_device_size() const { return [m_device recommendedMaxWorkingSetSize]; }
  // there are implicit allocations from MPS backend, so we need to query the 'device' for
  // total allocated size instead of manually tracking in MPSAllocator
  size_t current_allocated_size() const { return [m_device currentAllocatedSize]; }

  void trigger_memory_callbacks(BufferBlock* buffer_block, IMpsAllocatorCallback::EventType event) const {
    for (const auto& name : MPSAllocatorCallbacksRegistry()->Keys()) {
      MPSAllocatorCallbacksRegistry()->Create(name)->executeMPSAllocatorCallback(buffer_block->buffer, event);
    }
  }

  // TODO: make a common function to do size unit conversions in PyTorch.
  static std::string format_size(uint64_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024UL) { os << size << " bytes"; }
    else if (size <= 1048576UL) { os << ((float) size / 1024.0) << " KB"; }
    else if (size <= 1073741824UL) { os << ((float) size / 1048576.0) << " MB"; }
    else { os << ((float) size / 1073741824.0) << " GB"; }
    return os.str();
  }
};

} // namespace HeapAllocator

// interface exposed to internal MPS operations

// get the requested non-aligned size of an MTL buffer
ssize_t get_requested_buffer_size(void* ptr);
// retrieve the shape of a base tensor from a view tensor
IntArrayRef get_buffer_shape(void* ptr);
// set the shape of a base tensor from a view tensor
void set_buffer_shape(void* ptr, const IntArrayRef& shape);
// allocate a buffer from a specialized pool to import CPU scalars into GPU
DataPtr allocate_scalar_buffer(void* value, size_t size);

} // namespace mps
} // namespace at
