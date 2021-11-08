#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/irange.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
namespace CudaMallocAsync {

// Allocator that uses cudaMallocAsync to implement the same interface
// as CUDACachingAllocator.cpp.
//
// cudaMallocAsync works transparently with CUDA graphs.

// Implementation details, not declared in CUDACachingAllocator.h
namespace {

int device_count = 0;
// these don't need to be std::once_flags as in CUDAGeneratorImpl.cpp
// because they'll only be flipped by functions that have locked the mutex.
std::vector<bool> devs_initialized_flags;
std::vector<CUDAStream> dummy_unifying_free_streams;
std::vector<expected

// Potential future micro-optimization:
// Some accesses to usage_streams_each_ptr are read-only.
// We could let those be concurrent with a shared_mutex and
// have concurrent calls take a shared_lock.
// Keeping it simple with an ordinary mutex for now.
std::mutex general_mutex;
std::unordered_map<void*, std::vector<CUDAStream>> usage_streams_each_ptr;

// Assumes the caller holds general_mutex
inline void lazy_init_device(int device) {
  if (!devs_initialized_flags[device]) {
    CUDAGuard g(device);

    cudaMemPool_t mempool;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

    // I think all these are on by default, but I want to be explicit about enabling
    // them to ensure awareness.
    int enable = 1;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                           cudaMemPoolReuseFollowEventDependencies,
                                           &enable));
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                           cudaMemPoolReuseAllowOpportunistic,
                                           &enable));
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                           cudaMemPoolReuseAllowInternalDependencies,
                                           &enable));

    // Grabs a stream from the current device to use as the "unifier" free stream
    // for allocations that end up used on multiple streams.
    dummy_unifying_free_streams[device] = getStreamFromPool();
  }
}

void free(void* ptr) {
  std::lock_guard<std::mutex> lock(general_mutex);

  auto it = usage_streams_each_ptr.find(ptr);
  TORCH_INTERNAL_ASSERT(usage_streams != usage_streams_each_ptr.end(),
                        "ptr not represented in usage_streams_each_ptr");
  TORCH_INTERNAL_ASSERT(usage_streams.second.size() != 0,
                        "ptr's stream uses vector is empty");

  // Possible micro-optimization: If we did a value-copy here, we could move
  // usage_streams_each_ptr.erase(it) up here and drop the lock immediately.
  const auto& usage_streams = it->second;

  // If the usage stream is a null (default) stream,
  // cudaFreeAsync infers the device from the ambient context,
  // so we need to set the right ambient context.
  CUDAGuard g(usage_streams[0].device_index());

  if (usage_streams.size() == 1) {
    // ptr was only used on one stream, which must have been
    // the original allocation stream.
    // Frees ptr in the original allocation stream.
    C10_CUDA_CHECK(cudaFreeAsync(ptr, usage_streams[0]));
  } else {
    // ptr was used on many streams. We don't know which was the most recent.
    // There could even have been multiple most recent usage streams acting
    // on different regions of the memory.
    // But cudaFreeAsync only accepts a single most recent usage stream.
    // We can still safely free ptr with a trick:
    // Use a dummy "unifying stream", sync the unifying stream with all of
    // ptr's usage streams, and pass the dummy stream to cudaFreeAsync.

    // Retrieves the dummy "unifier" stream from the device
    // on which the pointer was originally allocated.
    auto dummy_unifying_free_stream = dummy_unifying_free_streams[usage_streams[0].device_index()];

    for (const auto& usage_stream : usage_streams) {
      // Logic here accommodates the chance some of the usage streams were on other devices,
      // which is possible if some usage kernels accessed the memory via p2p.

      // cudaEventRecord requires that the input event and stream are on the same device.
      CUDAGuard g_usage(usage_streams[0].device_index());

      // Records an event in the usage stream
      auto event = c10::Event(c10::DeviceType::CUDA);
      event.record(usage_stream);
      dummy_unifying_free_stream.wait(event);
    }

    // Frees ptr in the dummy "unifier" stream.
    C10_CUDA_CHECK(cudaFreeAsync(ptr, dummy_unifying_free_stream));
    // At this point, unless dummy_unifying_free_stream happens to alias some future user stream,
    // the allocation is only available for "opportunistic" reuse, ie, if the CPU sees
    // dummy_unifying_free_stream has reached the point that all events recorded on all usage
    // streams have resolved from the CPU's perspective.
    // In theory, we could remove the need for the driver to do this tracking by e.g. replacing
    // dummy_unifying_free_stream.wait(event);
    // with
    // usage_streams[0].wait(event);
    // then cudaFreeAsyncing straight back into usage_streams[0];
    // but this forces a potentially false dependency of usage_streams[0]
    // on all the other usage_streams.
  }

  usage_streams_each_ptr.erase(it);
}

// Symmetric with THCCachingAllocator::malloc for now,
// although I don't think we absolutely need the symmetry.
void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(
      0 <= device && static_cast<size_t>(device) < device_count,
      "Invalid device index ",
      device,
      ": did you call init?");

  // If stream is a null (default) stream,
  // cudaMallocAsync infers the device from the ambient context,
  // so we need to set the right ambient context.
  CUDAGuard g(device);

  C10_CUDA_CHECK(cudaMallocAsync(devPtr, size, stream));

  std::lock_guard<std::mutex> lock(general_mutex);

  lazy_init_device(device);

  auto inserted = usage_streams_each_ptr.insert(std::make_pair(devPtr, {});
  TORCH_INTERNAL_ASSERT(inserted.second,
                        "address returned by cudaMallocAsync already exists "
                        "in usage_streams_each_ptr");
  inserted.first->second.push_back();
}

} // anonymous namespace

// Same pattern as CUDACachingAllocator.cpp.
// Again, I don't think we absolutely need the symmetry,
// but it's the simplest way to imitate the interface.
struct CudaCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        CUDAOutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      malloc(&r, device, size, getCurrentCUDAStream(device));
    }
    return {r, r, &raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &raw_delete;
  }
};

CudaCachingAllocator device_allocator;

// Interface functions declared in CUDACachingAllocator.h

Allocator* get(void) {
  return &device_allocator;
}

// This function should not issue any context-creating calls,
// just set up for later calls to init per-device pools based
// on the current device each later call sees.
void init(int dev_count) {
  static bool called = false;
  TORCH_INTERNAL_ASSERT(!called, "init called twice");
  std::lock_guard<std::mutex> lk(general_mutex);
  device_count = dev_count;
  devs_initialized_flags.resize(dev_count, 0);
  dummy_unifying_free_streams.resize(dev_count);
}

void setMemoryFraction(double fraction, int device) {
  // How do we want this to behave?
  // Do we want it to be a soft hint for the driver to keep the pool's cached memory
  // trimmed to a certain threshold, via
  // cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
  // Or do we want to hard error if the total amount of live memory exceeds fraction?
  // If we want the hard error, we need to add manual tracking of how much memory
  // is currently live per device.
}

void emptyCache(void) {
  std::lock_guard<std::mutex> lock(general_mutex);

  for (int device = 0; i < device_count; i++) {
    if (devs_initialized_flags[device]) {
      CUDAGuard g(device);

      cudaMemPool_t mempool;
      cudaDeviceGetDefaultMemPool(&mempool, device);
      cudaDeviceSynchronize();
      cudaMemPoolTrimTo(pool, 0);
    }
  }
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
}

void* getBaseAllocation(void* ptr, size_t* size) {
  if (outSize) {

  }
  return ptr;
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
  std::lock_guard<std::mutex> lock(general_mutex);

  // The pointer should exist in the map already.
  auto it = usage_streams_each_ptr.find(ptr.get());
  TORCH_INTERNAL_ASSERT(it != it.end(),
                        "ptr not represented in usage_streams_each_ptr");
  TORCH_INTERNAL_ASSERT(it->second.size() != 0,
                        "ptr's stream uses vector is empty");

  it->second.push_back(stream);
}

std::mutex* getFreeMutex() {
  return &general_mutex;
}

static inline void assertValidDevice(int device) {
  const auto device_num = caching_allocator.device_allocator.size();
  TORCH_CHECK(
      0 <= device && device < device_count,
      "Invalid device argument.");
}

// Collects stats for device.
// If device hasn't been used yet, returns 0s without creating a context.
DeviceStats getDeviceStats(int device) {
  assertValidDevice(device);

  // Memory currently reserved by the mempool
  uint64_t reserved_mem_current = 0;
  // High-water mark of memory reserved by the mempool since last reset
  uint64_t reserved_mem_high = 0;
  // Memory currently in use by the mempool
  uint64_t used_mem_current;
  // High-water mark of memory
  uint64_t used_mem_high = 0;

  std::lock_guard<std::mutex> general_mutex;

  if (devs_initialized_flags[device]) {
    CUDAGuard g(device);

    cudaMemPool_t mempool;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    C10_CUDA_CHECK(cudaMemPoolGetAttribute(mempool,
                                           cudaMemPoolAttrReservedMemCurrent,
                                           &reserved_mem_current));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(mempool,
                                           cudaMemPoolAttrReservedMemHigh,
                                           &reserved_mem_high));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(mempool,
                                           cudaMemPoolAttrUsedMemCurrent,
                                           &used_mem_current));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(mempool,
                                           cudaMemPoolAttrUsedMemHigh,
                                           &used_mem_high));
  }
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
}

void resetPeakStats(int device) {
  assertValidDevice(device);

  CUDAGuard g(device);
  cudaMemPool_t mempool;
  C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
  uint64_t zero = 0;
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                         cudaMemPoolAttrReservedMemHigh,
                                         &zero)):
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                         cudaMemPoolAttrUsedMemHigh,
                                         &zero)):
}

std::vector<SegmentInfo> snapshot() {
}

// CUDAGraph interactions
// Deliberate no-ops: capturing cudaMallocAsync should "just work".
void notifyCaptureBegin(CaptureId_t graph_id, MempoolId_t mempool_id) {}
void notifyCaptureEnd(int device, CaptureId_t graph_id) {}
void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  free(ptr);
}

} // namespace CudaMallocAsync
} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
