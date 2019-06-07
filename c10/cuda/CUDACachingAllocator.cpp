#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda {
namespace CUDACachingAllocator {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
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



namespace {

using stream_set = std::unordered_set<cuda::CUDAStream>;

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB

struct DeviceStats {
  uint64_t   amount_allocated;      // total amount allocated in bytes
  uint64_t   max_amount_allocated;  // max total amount allocated in bytes
  uint64_t   amount_cached;         // total amount in cache in bytes
  uint64_t   max_amount_cached;     // max total amount in cache in bytes

  DeviceStats() :
      amount_allocated(0), max_amount_allocated(0),
      amount_cached(0), max_amount_cached(0) { }

  void increaseAllocated(size_t delta) {
    amount_allocated += delta;
    max_amount_allocated = std::max(max_amount_allocated, amount_allocated);
  }

  void decreaseAllocated(size_t delta) {
    amount_allocated -= delta;
  }

  void increaseCached(size_t delta) {
    amount_cached += delta;
    max_amount_cached = std::max(max_amount_cached, amount_cached);
  }

  void decreaseCached(size_t delta) {
    amount_cached -= delta;
  }
};

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
typedef std::set<Block*, Comparison> BlockPool;

struct Block {
  int           device;      // gpu
  cudaStream_t  stream;      // allocation stream
  stream_set    stream_uses; // streams on which the block was used
  size_t        size;        // block size in bytes
  BlockPool*    pool;        // owning memory pool
  void*         ptr;         // memory address
  bool          allocated;   // in-use flag
  Block*        prev;        // prev block if split from a larger allocation
  Block*        next;        // next block if split from a larger allocation
  int           event_count; // number of outstanding CUDA events

  Block(int device, cudaStream_t stream, size_t size, BlockPool* pool, void* ptr) :
    device(device), stream(stream), stream_uses(), size(size), pool(pool),
    ptr(ptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size) :
    device(device), stream(stream), stream_uses(), size(size), pool(nullptr),
    ptr(nullptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }
};

static bool BlockComparator(const Block* a, const Block* b)
{
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
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

} // namespace

struct THCCachingAllocator
{
  // device statistics
  std::vector<DeviceStats> device_stats;

  // lock around all operations
  std::recursive_mutex mutex;

  // lock around calls to cudaFree (to prevent deadlocks with NCCL)
  std::mutex cuda_free_mutex;

  // cached blocks larger than 1 MB
  BlockPool large_blocks;

  // cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events;

  THCCachingAllocator() :
      large_blocks(BlockComparator),
      small_blocks(BlockComparator) {}

  DeviceStats &get_stats_for_device(int device) {
    AT_ASSERT(device >= 0);
    if ((size_t) device >= device_stats.size()) {
      device_stats.resize(device + 1);
    }
    return device_stats.at(device);
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, size_t size, cudaStream_t stream)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    // process outstanding cudaEvents
    process_events();

    size = round_size(size);

    DeviceStats &stats = get_stats_for_device(device);

    Block search_key(device, stream, size);
    auto& pool = get_pool(size);

    auto find_free_block = [&]()->Block*{
      auto it = pool.lower_bound(&search_key);
      if (it != pool.end() && (*it)->device == device &&
          (*it)->stream == stream) {
        Block* block = *it;
        pool.erase(it);
        return block;
      }
      return nullptr;
    };

    Block* block = find_free_block();
    if (block == nullptr) {
      bool freed_memory = false;
      for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
        freed_memory |=
            FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
      }
      if (freed_memory) {
        block = find_free_block();
      }
    }
    if (block == nullptr) {
      void* ptr;
      size_t alloc_size = get_allocation_size(size);
      cudaError_t err = cuda_malloc_retry(device, &ptr, alloc_size);
      if (err != cudaSuccess) {
        if (err == cudaErrorMemoryAllocation) {
          cudaGetLastError();  // clear CUDA error

          size_t device_free;
          size_t device_total;
          C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
          const auto& stats = get_stats_for_device(device);

          // "total capacity": total global memory on GPU
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
          // Note that at this point cuda_malloc_retry has already returned all
          // possible "cached" memory to the driver. The only remaining "cached"
          // memory is split from a larger block that is partially in-use.
          AT_ERROR(
            "CUDA out of memory. Tried to allocate ", format_size(alloc_size),
            " (GPU ", device, "; ",
            format_size(device_total), " total capacity; ",
            format_size(stats.amount_allocated), " already allocated; ",
            format_size(device_free), " free; ",
            format_size(stats.amount_cached - stats.amount_allocated), " cached)");
        } else {
          C10_CUDA_CHECK(err);
        }
      }
      stats.increaseCached(alloc_size);
      block = new Block(device, stream, alloc_size, &pool, ptr);
    }

    Block* remaining = nullptr;
    AT_ASSERT(block);
    if (should_split(block, size)) {

      remaining = block;

      block = new Block(device, stream, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.insert(remaining);
    }

    block->allocated = true;
    allocated_blocks[block->ptr] = block;

    *devPtr = block->ptr;

    stats.increaseAllocated(block->size);
  }

  void free(void* ptr)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    Block* block = it->second;
    allocated_blocks.erase(it);
    block->allocated = false;

    get_stats_for_device(block->device).decreaseAllocated(block->size);
    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  /** returns cached blocks to the system allocator */
  void emptyCache()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    synchronize_and_free_events(nullopt);
    free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
    free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
  }

  void* getBaseAllocation(void* ptr, size_t* outSize)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    Block* block = find_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
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

  // Accumulates sizes of all memory blocks for given device in given pool
  void cacheInfoAux(BlockPool& blocks, int dev_id, size_t* total, size_t* largest)
  {
    Block search_key(dev_id, 0, 0);
    auto it = blocks.lower_bound(&search_key);
    for (; it != blocks.end() && *it && (*it)->device == dev_id; ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void cacheInfo(int dev_id, size_t* total, size_t* largest)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cacheInfoAux(large_blocks, dev_id, total, largest);
    cacheInfoAux(small_blocks, dev_id, total, largest);
  }

  void recordStream(void* ptr, cuda::CUDAStream stream)
  {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (ptr) {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      Block* block = find_allocated_block(ptr);
      if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
      }
      if (stream.stream() == block->stream) {
        // ignore uses on the allocation stream, since those don't require any
        // special synchronization
        return;
      }
      block->stream_uses.insert(stream);
    }
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block)
  {
    AT_ASSERT(!block->allocated && block->event_count == 0);
    auto& pool = *block->pool;
    try_merge_blocks(block, block->prev, pool);
    try_merge_blocks(block, block->next, pool);
    pool.insert(block);
  }

  /** combine previously split blocks */
  void try_merge_blocks(Block* dst, Block* src, BlockPool& pool)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    pool.erase(src);
    delete src;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool should_split(Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  cudaError_t cuda_malloc_retry(int device, void** devPtr, size_t size)
  {
    // Try cudaMalloc. If cudaMalloc fails, frees all non-split cached blocks
    // and retries.
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess) {
      cudaGetLastError();  // reset the last CUDA error
      free_cached_blocks(device);
      err = cudaMalloc(devPtr, size);
      if (err != cudaSuccess) {
        return err;
      }
    }
    return cudaSuccess;
  }

  void free_cached_blocks(int device)
  {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(device);

    // Free all non-split cached blocks on device
    Block lower_bound(device, nullptr, 0);
    Block upper_bound(device + 1, nullptr, 0);

    free_blocks(
        large_blocks,
        large_blocks.lower_bound(&lower_bound),
        large_blocks.lower_bound(&upper_bound));
    free_blocks(
        small_blocks,
        small_blocks.lower_bound(&lower_bound),
        small_blocks.lower_bound(&upper_bound));
  }

  void free_blocks(BlockPool& blocks, BlockPool::iterator it, BlockPool::iterator end)
  {
    // Frees all non-split blocks between `it` and `end`
    std::lock_guard<std::mutex> lock(cuda_free_mutex);
    while (it != end) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        C10_CUDA_CHECK(cudaFree((void*)block->ptr));
        get_stats_for_device(block->device).decreaseCached(block->size);
        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  void synchronize_and_free_events(optional<int> device) {
    // Synchronize on outstanding events and then free associated blocks.
    // Limited to blocks on the given device if specified.

    auto remaining_events = decltype(cuda_events)();

    for (auto& e : cuda_events) {
      cudaEvent_t event = e.first;
      Block* block = e.second;
      if (device.has_value() && block->device != *device) {
        remaining_events.push_back(e);
        continue;
      }

      C10_CUDA_CHECK(cudaEventSynchronize(event));
      C10_CUDA_CHECK(cudaEventDestroy(event));

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    std::swap(cuda_events, remaining_events);
  }

  Block* find_allocated_block(void *ptr) {
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    return it->second;
  }

  void insert_events(Block* block)
  {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      C10_CUDA_CHECK(cudaSetDevice(it->device_index()));

      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(event, it->stream()));

      block->event_count++;
      cuda_events.emplace_back(event, block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void process_events()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      cudaEvent_t event = e.first;
      Block* block = e.second;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        // ignore and clear the error if not ready
        cudaGetLastError();
        break;
      } else if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
      }

      C10_CUDA_CHECK(cudaEventDestroy(event));

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      cuda_events.pop_front();
    }
  }
};

THCCachingAllocator caching_allocator;

static void CudaCachingDeleter(void* ptr) {
  caching_allocator.free(ptr);
}

// NB: I decided not to fold this into THCCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publically exposed
struct CudaCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      caching_allocator.malloc(&r, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &CudaCachingDeleter, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &CudaCachingDeleter;
  }
};

CudaCachingAllocator device_allocator;

Allocator* get(void)
{
  return &device_allocator;
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.cacheInfo(dev_id, cachedAndFree, largestBlock);
}

void* getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(void *ptr, cuda::CUDAStream stream)
{
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex()
{
  return &caching_allocator.cuda_free_mutex;
}

static inline void assertValidDevice(int device) {
  int device_num = device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

uint64_t currentMemoryAllocated(int device)
{
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_allocated;
}

uint64_t maxMemoryAllocated(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_allocated;
}

void resetMaxMemoryAllocated(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_allocated = stats.amount_allocated;
}

uint64_t currentMemoryCached(int device)
{
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_cached;
}

uint64_t maxMemoryCached(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_cached;
}

void resetMaxMemoryCached(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_cached = stats.amount_cached;
}

//
// In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
// is called by the receiving process to map the CUDA memory from the sending
// process into its own address space.
//
// CUDA IPC only allows sharing a big memory block associated with a cudaIpcMemHandle_t
// and it can be opened only **once** per context per process. There can be
// multiple types of storage in the same IPC mem block, so we must cache the
// device ptr to construct typed storage as it comes.
//
// ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the process
// that can be used to access the memory block in the sender process.
// It only saves a weak_ptr of the device pointer in the map, the shared_ptr
// will be used to reconstruct all storages in this CudaMalloc allocation.
// And it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
//
namespace {
  std::mutex IpcMutex;
  std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
}

std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  std::lock_guard<std::mutex> lock(IpcMutex);

  auto iter = ipcMemHandle_to_devptr.find(handle);
  if (iter != ipcMemHandle_to_devptr.end()) {
    auto devptr = iter->second.lock();
    if (devptr) return devptr;
  }
  // This ipcMemHandle hasn't been opened, or already expired, open it to
  // enable IPC access to that mem block.
  void *dev = nullptr;
  auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  // devPtr has to be deleted in same device when created.
  int curr_device;
  C10_CUDA_CHECK(cudaGetDevice(&curr_device));
  auto sp = std::shared_ptr<void>(
      dev,
      [handle, curr_device](void *ptr) {
        cuda::CUDAGuard device_guard(curr_device);
        std::lock_guard<std::mutex> deleter_lock(IpcMutex);
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
        ipcMemHandle_to_devptr.erase(handle);});
  std::weak_ptr<void> wp = sp;
  // To eliminate an additional search, we can use insert().
  // It doesn't overwrite when key already exists(ptr expired).
  // But in the deleter for sp we erased the entry,
  // this should be safe to do now.
  ipcMemHandle_to_devptr.insert(iter, {handle, wp});

  return sp;
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

} // namespace CUDACachingAllocator

}} // namespace c10::cuda
