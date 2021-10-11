#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Registry.h>

#include <array>
#include <mutex>

namespace c10 {

class C10_CUDA_API CUDAOutOfMemoryError : public c10::Error {
  using Error::Error;
};

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class C10_CUDA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeCudaMemoryCallbacksRegistry, name, __VA_ARGS__);

namespace cuda {

// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (CUDACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THCCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.

namespace CUDACachingAllocator {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from cudaMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via cudaFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;

  // COUNT: total number of failed calls to CUDA malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  bool allocated = false;
  bool active = false;
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  std::vector<BlockInfo> blocks;
};

// Deliberately not marked with C10_CUDA_API visibility to external libs.
#define CUDA_ALLOCATOR_BACKEND_INTERFACE \
void* raw_alloc(size_t nbytes); \
void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream); \
void raw_delete(void* ptr); \
Allocator* get(); \
void init(int device_count); \
void setMemoryFraction(double fraction, int device); \
void emptyCache(); \
void cacheInfo( int dev_id, size_t* cachedAndFree, size_t* largestBlock); \
void* getBaseAllocation(void* ptr, size_t* size); \
void recordStream(const DataPtr&, CUDAStream stream); \
DeviceStats getDeviceStats(int device); \
void resetAccumulatedStats(int device); \
void resetPeakStats(int device); \
std::vector<SegmentInfo> snapshot(); \
void notifyCaptureBegin( int device, CaptureId_t graph_id, MempoolId_t mempool_id); \
void notifyCaptureEnd(int device, CaptureId_t graph_id); \
void notifyCaptureDestroy(int device, MempoolId_t mempool_id); \
std::mutex* getFreeMutex();

// Not meant to be called directly by clients.
// Maybe make "CUDACachingAllocator" a class or struct, and make these private members?
namespace THC {
CUDA_ALLOCATOR_BACKEND_INTERFACE
}

// Not meant to be called directly by clients.
namespace CudaMallocAsync {
CUDA_ALLOCATOR_BACKEND_INTERFACE
}

// Returns string describing the allocator backend
// (currently "native" or "cudaMallocAsync")
std::string allocatorBackend();

// The following functions ARE meant to be called directly by clients.
// They'll choose the appropriate backend based on the runtime value of
// the PYTORCH_CUDA_ALLOC_CONF environment variable
// (cf CachingAllocatorConfig in CUDACachingAllocator.cpp)

C10_CUDA_API void* raw_alloc(size_t nbytes) {
  // Lean on the out-of-line call here as the surface at which we choose the allocator backend
  // The selecting conditional could easily be made smarter, ie, it could be a lambda like
  // static auto f = [] {
  //   const std::string backend = allocatorBackend();
  //   if (backend == "native") {
  //     return CudaMallocAsync::raw_alloc;
  //   else if (backend == "cudaMallocAsync") {
  //     return CudaMallocAsync::raw_alloc;
  //   else {
  //     Assume "backend" is the name of a user-supplied .so.
  //     Use dlopen+dlsym to fish the raw_alloc symbol from it.
  //     return raw_alloc_symbol_from_user_lib;
  //   }();
  static auto f = (allocatorBackend() == "native") ?
    THC::raw_alloc : cudaMallocAsync::raw_alloc;
  return f(nbytes);
  // The downside of ^ is that it's not inlineable even with LTO.
  // If LTO is enabled and we think it's capable of inlining any allocator backend's calls at the
  // point of use, we could just use if statements and hope for good branch prediction.
  // static bool useNative = (CachingAllocatorConfig::allocator_backend() == "native");
  // if (useCudaMallocAsync) {
  //   return CudaMallocAsync(raw_alloc);
  // } else {
  //   return THC::raw_alloc(nbytes);
  // }
}

C10_CUDA_API void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  static auto f = (allocatorBackend() == "native") ?
    THC::raw_alloc_with_stream : cudaMallocAsync::raw_alloc_with_stream;
  return f(nbytes, stream);
}

C10_CUDA_API void raw_delete(void* ptr) {
  static auto f = (allocatorBackend() == "native") ?
    THC::raw_delete : cudaMallocAsync::raw_delete;
  return f(ptr);
}

C10_CUDA_API Allocator* get() {
  static auto f = (allocatorBackend() == "native") ?
    THC::get : cudaMallocAsync::get;
  return f();
}

C10_CUDA_API void init(int device_count) {
  static auto f = (allocatorBackend() == "native") ?
    THC::init : cudaMallocAsync::init;
  return f(device_count);
}

C10_CUDA_API void setMemoryFraction(double fraction, int device) {
  static auto f = (allocatorBackend() == "native") ?
    THC::setMemoryFraction : cudaMallocAsync::setMemoryFraction;
  f(fraction, device);
}

C10_CUDA_API void emptyCache() {
  static auto f = (allocatorBackend() == "native") ?
    THC::empty_cache : cudaMallocAsync::empty_cache;
  return f();
}

C10_CUDA_API void cacheInfo(
    int dev_id,
    size_t* cachedAndFree,
    size_t* largestBlock) {
  static auto f = (allocatorBackend() == "native") ?
    THC::cacheInfo : cudaMallocAsync::cacheInfo;
  return f(dev_id, cachedAndFree, largestBlock);
}

C10_CUDA_API void* getBaseAllocation(void* ptr, size_t* size) {
  static auto f = (allocatorBackend() == "native") ?
    THC::getBaseAllocation : cudaMallocAsync::getBaseAllocation;
  return f(ptr, size);
}

C10_CUDA_API void recordStream(const DataPtr& dataPtr, CUDAStream stream) {
  static auto f = (allocatorBackend() == "native") ?
    THC::recordStream : cudaMallocAsync::recordStream;
  return f(dataPtr, stream);
}

C10_CUDA_API DeviceStats getDeviceStats(int device) {
  static auto f = (allocatorBackend() == "native") ?
    THC::getDeviceStats : cudaMallocAsync::getDeviceStats;
  return f(device);
}

C10_CUDA_API void resetAccumulatedStats(int device) {
  static auto f = (allocatorBackend() == "native") ?
    THC::resetAccumulatedStats : cudaMallocAsync::resetAccumulatedStats;
  return f(device);
}

C10_CUDA_API void resetPeakStats(int device) {
  static auto f = (allocatorBackend() == "native") ?
    THC::resetPeakStats : cudaMallocAsync::resetPeakStats;
  return f(device);
}

C10_CUDA_API std::vector<SegmentInfo> snapshot() {
  static auto f = (allocatorBackend() == "native") ?
    THC::snapshot : cudaMallocAsync::snapshot;
  return f();
}

// CUDAGraph interactions
C10_CUDA_API void notifyCaptureBegin(
    int device,
    CaptureId_t graph_id,
    MempoolId_t mempool_id) {
  static auto f = (allocatorBackend() == "native") ?
    THC::notifyCaptureBegin : cudaMallocAsync::notifyCaptureBegin;
  return f(device, graph_id, mempool_id);
}

C10_CUDA_API void notifyCaptureEnd(int device, CaptureId_t graph_id) {
  static auto f = (allocatorBackend() == "native") ?
    THC::notifyCaptureEnd : cudaMallocAsync::notifyCaptureEnd;
  return f(device, graph_id);
}
C10_CUDA_API void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {
  static auto f = (allocatorBackend() == "native") ?
    THC::notifyCaptureDestroy : cudaMallocAsync::notifyCaptureDestroy;
  return f(device, mempool_id);
}

C10_CUDA_API std::mutex* getFreeMutex() {
  static auto f = (allocatorBackend() == "native") ?
    THC::getFreeMutex : cudaMallocAsync::getFreeMutex;
  return f();
}

// Not part of CUDA_ALLOCATOR_BACKEND_INTERFACE
C10_CUDA_API std::shared_ptr<void> getIpcDevPtr(std::string handle);

} // namespace CUDACachingAllocator

} // namespace cuda
} // namespace c10

#endif
