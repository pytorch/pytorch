#pragma once

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
  int32_t gc_counter = 0;
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

// Allocator config options.
enum struct AllocatorBackend : uint8_t {
  NATIVE = 0,
  CUDAMALLOCASYNC = 1,
};

C10_CUDA_API AllocatorBackend allocatorBackend();

// Size pretty-printer
std::string format_size(uint64_t size);

#define CUDA_ALLOCATOR_BACKEND_INTERFACE \
C10_CUDA_API void* raw_alloc(size_t nbytes); \
C10_CUDA_API void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream); \
C10_CUDA_API void raw_delete(void* ptr); \
C10_CUDA_API Allocator* get(); \
C10_CUDA_API void init(int device_count); \
C10_CUDA_API void setMemoryFraction(double fraction, int device); \
C10_CUDA_API void emptyCache(); \
C10_CUDA_API void cacheInfo(int dev_id, size_t* largestBlock); \
C10_CUDA_API void* getBaseAllocation(void* ptr, size_t* size); \
C10_CUDA_API void recordStream(const DataPtr&, CUDAStream stream); \
C10_CUDA_API DeviceStats getDeviceStats(int device); \
C10_CUDA_API void resetAccumulatedStats(int device); \
C10_CUDA_API void resetPeakStats(int device); \
C10_CUDA_API std::vector<SegmentInfo> snapshot(); \
C10_CUDA_API void notifyCaptureBegin(int device, CaptureId_t graph_id, MempoolId_t mempool_id); \
C10_CUDA_API void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id); \
C10_CUDA_API void notifyCaptureEnded(int device, CaptureId_t graph_id); \
C10_CUDA_API void notifyCaptureDestroy(int device, MempoolId_t mempool_id); \
C10_CUDA_API std::mutex* getFreeMutex(); \
C10_CUDA_API std::shared_ptr<void> getIpcDevPtr(std::string handle);

// Not meant to be called directly by clients.
// Maybe make "CUDACachingAllocator" a class or struct, and make these private members?
namespace Native {
CUDA_ALLOCATOR_BACKEND_INTERFACE
}

// Not meant to be called directly by clients.
namespace CudaMallocAsync {
CUDA_ALLOCATOR_BACKEND_INTERFACE
}

// The following functions ARE meant to be called directly by clients.
// They'll choose the appropriate backend based on the runtime value of
// the PYTORCH_CUDA_ALLOC_CONF environment variable
// (cf parseArgs in CUDACachingAllocator.cpp)

inline void* raw_alloc(size_t nbytes) {
  // Lean on the out-of-line call here as the surface at which we choose the allocator backend
  // The selecting conditional could easily be made smarter, ie, it could be a lambda like
  // static auto f = [] {
  //     if (allocatorBackend() == AllocatorBackend::NATIVE) {
  //       return CudaMallocAsync::raw_alloc;
  //     } else if (backend == "cudaMallocAsync") {
  //       return CudaMallocAsync::raw_alloc;
  //     } else {
  //       Assume "backend" is the name of a user-supplied .so.
  //       Use dlopen+dlsym to fish the raw_alloc symbol from it.
  //       return raw_alloc_symbol_from_user_lib;
  //     }
  //   }();
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::raw_alloc : CudaMallocAsync::raw_alloc;
  return f(nbytes);
  // The downside of ^ is that it's not inlineable even with LTO.
  // If LTO is enabled and we think it's capable of inlining any allocator backend's calls at the
  // point of use, we could just use if statements and hope for good branch prediction.
  // static bool useNative = (allocatorBackend() == "native");
  // if (useCudaMallocAsync) {
  //   return CudaMallocAsync(raw_alloc);
  // } else {
  //   return Native::raw_alloc(nbytes);
  // }
}

inline void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::raw_alloc_with_stream : CudaMallocAsync::raw_alloc_with_stream;
  return f(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::raw_delete : CudaMallocAsync::raw_delete;
  return f(ptr);
}

inline Allocator* get() {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::get : CudaMallocAsync::get;
  return f();
}

inline void init(int device_count) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::init : CudaMallocAsync::init;
  return f(device_count);
}

inline void setMemoryFraction(double fraction, int device) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::setMemoryFraction : CudaMallocAsync::setMemoryFraction;
  f(fraction, device);
}

inline void emptyCache() {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::emptyCache : CudaMallocAsync::emptyCache;
  return f();
}

inline void cacheInfo(int dev_id, size_t* largestBlock) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::cacheInfo : CudaMallocAsync::cacheInfo;
  return f(dev_id, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::getBaseAllocation : CudaMallocAsync::getBaseAllocation;
  return f(ptr, size);
}

inline void recordStream(const DataPtr& dataPtr, CUDAStream stream) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::recordStream : CudaMallocAsync::recordStream;
  return f(dataPtr, stream);
}

inline DeviceStats getDeviceStats(int device) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::getDeviceStats : CudaMallocAsync::getDeviceStats;
  return f(device);
}

inline void resetAccumulatedStats(int device) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::resetAccumulatedStats : CudaMallocAsync::resetAccumulatedStats;
  return f(device);
}

inline void resetPeakStats(int device) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::resetPeakStats : CudaMallocAsync::resetPeakStats;
  return f(device);
}

inline std::vector<SegmentInfo> snapshot() {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::snapshot : CudaMallocAsync::snapshot;
  return f();
}

// CUDAGraph interactions
inline void notifyCaptureBegin(
    int device,
    CaptureId_t graph_id,
    MempoolId_t mempool_id) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::notifyCaptureBegin : CudaMallocAsync::notifyCaptureBegin;
  return f(device, graph_id, mempool_id);
}

inline void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::notifyCaptureAboutToEnd : CudaMallocAsync::notifyCaptureAboutToEnd;
  return f(device, graph_id);
}

inline void notifyCaptureEnded(int device, CaptureId_t graph_id) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::notifyCaptureEnded : CudaMallocAsync::notifyCaptureEnded;
  return f(device, graph_id);
}

inline void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::notifyCaptureDestroy : CudaMallocAsync::notifyCaptureDestroy;
  return f(device, mempool_id);
}

inline std::mutex* getFreeMutex() {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::getFreeMutex : CudaMallocAsync::getFreeMutex;
  return f();
}

// Not part of CUDA_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  static auto f = (allocatorBackend() == AllocatorBackend::NATIVE) ?
    Native::getIpcDevPtr : CudaMallocAsync::getIpcDevPtr;
  return f(handle);

}

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
