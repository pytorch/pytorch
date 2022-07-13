//  Copyright © 2022 Apple Inc.

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <c10/util/flat_hash_map.h>

#include <ATen/mps/MPSDevice.h>
#include <cstdio>
#include <mutex>
#include <set>
#include <utility>
#include <mach/vm_page_size.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <Metal/MTLHeap.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

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

// TODO: check the caching performance of write-combined mode
constexpr MTLResourceOptions kCPUCacheMode = MTLResourceOptionCPUCacheModeDefault;
constexpr MTLResourceOptions kPrivateResourceOptions = kCPUCacheMode | MTLResourceStorageModePrivate;
constexpr MTLResourceOptions kSharedResourceOptions  = kCPUCacheMode | MTLResourceStorageModeShared;

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

  static MTLResourceOptions getOptions(bool SharedStorage = false) { return SharedStorage ? kSharedResourceOptions : kPrivateResourceOptions; }

  static id<MTLHeap> createMTLHeap(id<MTLDevice> device, size_t size, bool is_shared) {
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
      d.storageMode = is_shared ? MTLStorageModeShared : MTLStorageModePrivate;
      d.cpuCacheMode = MTLCPUCacheModeDefaultCache;
      // this automatically handles Metal buffer access synchronizations at the
      // cost of slightly lower performance.
      d.hazardTrackingMode = MTLHazardTrackingModeTracked;
      d.resourceOptions = getOptions(is_shared) | (MTLHazardTrackingModeTracked << MTLResourceHazardTrackingModeShift);
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
  id<MTLBuffer> newMTLBuffer(size_t length, bool is_shared) {
    id<MTLBuffer> buf = [heap newBufferWithLength:length options:getOptions(is_shared)];
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
  BufferPool(const id<MTLDevice> Device, bool Small, bool Shared) :
           device(Device), is_small(Small), is_shared(Shared),
           heaps(HeapBlock::Comparator), buffers(BufferBlock::Comparator) { }

  const id<MTLDevice> device;
  // small heaps have sizes of kSmallHeap, and large ones kLargeHeap
  const bool is_small;
  // private pools allocated on device memory; otherwise, shared between host/device
  const bool is_shared;
  // list of heaps ordered by their "available" (not total) memory size
  std::set<HeapBlock*, HeapComparison> heaps;
  // list of only "available" buffers in the pool (i.e., buffers not in-use)
  std::set<BufferBlock*, BufferComparison> buffers;
};

struct AllocParams
{
  AllocParams(size_t Alloc_Size, size_t Requested_Size, BufferPool* Pool) :
            search_key(Alloc_Size), pool(Pool),
            buffer_block(nullptr), requested_size(Requested_Size) {}
  size_t size() const { return search_key.size; }

  BufferBlock search_key;
  BufferPool* pool;
  BufferBlock* buffer_block;
  size_t requested_size;
};

class MPSHeapAllocatorImpl
{
public:
  explicit MPSHeapAllocatorImpl() :
                      m_device(at::mps::MPSDevice::getInstance()->device()),
                      m_large_pool_shared(m_device, false, true), m_large_pool_private(m_device, false, false),
                      m_small_pool_shared(m_device, true , true), m_small_pool_private(m_device, true , false),
                      m_total_allocated_memory(0), m_max_buffer_size([m_device maxBufferLength]),
                      m_set_fraction(false), m_enable_debug_info(false) { }

  // interface exposed to at::Allocator
  id<MTLBuffer> Malloc(size_t size, bool sharedStorage);
  void Free(void* ptr);
  void EmptyCache();
  bool isSharedBuffer(void* ptr);
  ssize_t getRequestedBufferSize(void* ptr);
  void setBufferShape(void* ptr, const IntArrayRef& shape);
  IntArrayRef getBufferShape(void* ptr);

  inline id<MTLDevice> Device() const { return m_device; }
  void enable_debug_info() { m_enable_debug_info = true; }
  bool debug_info_enabled() const { return m_enable_debug_info; }
  void set_shared_storage_mode(bool useSharedStorage);

private:
  const id<MTLDevice> m_device;
  std::mutex m_mutex;
  // allocated buffers by device pointer
  ska::flat_hash_map<void*, BufferBlock*> m_allocated_buffers;
  // unallocated cached buffers larger than 1 MB
  BufferPool m_large_pool_shared, m_large_pool_private;
  // unallocated cached buffers 1 MB or smaller
  BufferPool m_small_pool_shared, m_small_pool_private;
  // total memory allocated by HeapAllocator
  size_t m_total_allocated_memory;
  // max buffer size allowed by Metal
  size_t m_max_buffer_size;
  // sets a soft upper bound to limit the total allocations
  bool m_set_fraction;
  // use "PYTORCH_DEBUG_MPS_ALLOCATOR" env-var to enable debug info
  bool m_enable_debug_info;

  HeapBlock* get_free_heap(AllocParams& p);
  bool get_free_buffer(AllocParams& p);
  BufferBlock* get_allocated_buffer_block(void* ptr);
  bool alloc_buffer(AllocParams& p);
  void free_buffer(BufferBlock* buffer_block);
  void release_buffer(BufferBlock* buffer_block, bool remove_empty_heap = true);
  void release_buffers(BufferPool& pool);
  bool release_available_cached_buffers(const AllocParams& p);
  bool release_cached_buffers();
  void trigger_memory_callbacks(BufferBlock* buffer_block, IMpsAllocatorCallback::EventType event);

  BufferPool& get_pool(size_t Size, bool useShared) {
      return Size <= kMaxSmallAlloc ? (useShared ? m_small_pool_shared : m_small_pool_private) :
                                      (useShared ? m_large_pool_shared : m_large_pool_private);
  }

  size_t get_allocation_size(size_t Length, bool useShared) {
    MTLSizeAndAlign sizeAlign = [m_device heapBufferSizeAndAlignWithLength:Length
                                                                   options:HeapBlock::getOptions(useShared)];
    return BufferBlock::alignUp(sizeAlign.size, sizeAlign.align);
  }
  // TODO: make this configurable
  static size_t max_split_size() { return std::numeric_limits<size_t>::max(); }
  // maximum size of device memory available for allocation in current process
  size_t max_available_size() const { return [m_device recommendedMaxWorkingSetSize] - [m_device currentAllocatedSize]; }

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

} // namespace mps
} // namespace at
