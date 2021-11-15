#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/irange.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>
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

// General helpers

int device_count = 0;
// these don't need to be std::once_flags as in CUDAGeneratorImpl.cpp
// because they'll only be flipped by functions that have locked the mutex.
std::vector<bool> devs_initialized_flags;
std::vector<CUDAStream> dummy_unifying_free_streams;

// Potential future micro-optimization:
// Some accesses to usage_streams_each_ptr are read-only.
// We could let those be concurrent with a shared_mutex and
// have concurrent calls take a shared_lock.
// Keeping it simple with an ordinary mutex for now.
std::mutex general_mutex;

/**
 * Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During CUDA graph capture, it's illegal to call cudaFreeAsync
 * on a pointer that came from a non-captured cudaMallocAsync.
 * Unfortunately, Python being what it is, it's impossible to be
 * sure no uncaptured tensor will ever have its destructor called
 * in a capturing region.
 * We avoid errors by
 *  1. tracking captured and uncaptured allocated pointers separately
 *  2. during capture, if we detect an attempt to free an uncaptured
 *     allocation on a capturing stream, don't free it immediately,
 *     just remember it and defer its cudaFreeAsync call to after
 *     the end of capture (specifically, to notifyCaptureEnded).
 */
std::unordered_map<void*, std::vector<CUDAStream>> usage_streams_ungraphed_ptrs;
std::unordered_map<void*, std::vector<CUDAStream>> usage_streams_graphed_ptrs;
std::vector<void*> ungraphed_ptrs_defer_free_until_no_capture;

// Graph-specific helpers

/**
 * Note [Avoid dangling free streams during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During capture, all stream dependencies must branch out from
 * the stream on which capture began and rejoin this initial stream
 * before capture ends.
 * The user rigs desired forking and joining with event waits.
 * But it's hard to be sure when tensor destructors get called relative
 * to the final joins.
 * For example, suppose a user
 *   forks work stream B from initial capture stream A
 *   creates a tensor T in B
 *   joins by syncing A with B
 *   ends capture.
 * All well and good, right? Maybe not: maybe T went out of scope
 * and its destructor got called AFTER the rejoin, leaving the graph with
 * "unjoined work": a dangling cudaFreeAsync node in stream B.
 * Ensuring that all tensor destructors for all side stream tensors
 * are called before side streams rejoin the main stream is
 * difficult. The user might have to add a bunch of explicit
 * "del"s at the right spots in code that was fine for ordinary
 * eager execution.
 * Fortunately, we can spare the user this burden:
 * during capture, we remember _all_ free streams,
 * and manually rejoin them with the capture stream during
 * notifyCaptureAboutToEnd.
 * This approach is heavy-handed, but hopefully capture only needs to
 * happen once, so we don't mind being heavy-handed.
 *
 * TODO: If, someday, we augment the graph bindings to support recapture
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#whole-graph-update
 * (eg, as a way to accommodate dynamic params) we should think more
 * carefully about the CPU overhead of remembering and rejoining
 * all free streams during capture. Maybe it's not a big deal.
 */
std::unordered_set<CUDAStream> capture_free_streams;
bool capture_underway = false;

// Implementation functions

// Assumes the caller holds general_mutex
inline void lazy_init_device(int device) {
  if (!devs_initialized_flags[device]) {
    CUDAGuard g(device);

    cudaMemPool_t mempool;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

    // I think all these are on by default, but I want to enable them
    // explicitly to ensure awareness.
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
  std::lock_guard<std::mutex> lk(general_mutex);

  auto it = usage_streams_each_ptr.find(ptr);
  TORCH_INTERNAL_ASSERT(it != usage_streams_each_ptr.end(),
                        "ptr not represented in usage_streams_each_ptr");
  TORCH_INTERNAL_ASSERT(it->second.size() != 0,
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

    if (C10_UNLIKELY(captures_underway)) {
      // See Note [Avoid dangling free streams during CUDA graph capture]
      capture_free_streams.insert(usage_streams[0]);
    }
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

    // The number of usage streams is typically small (low single digits)
    for (const auto& usage_stream : usage_streams) {
      // Logic here accommodates the chance some of the usage streams were on other devices,
      // which is possible if some usage kernels accessed the memory via p2p.

      // cudaEventRecord requires that the input event and stream are on the same device.
      CUDAGuard g_usage(usage_stream.device_index());

      // CUDACachingAllocator.cpp uses raw cuda events, as do we.
      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(event, usage_stream.stream()));
      C10_CUDA_CHECK(cudaStreamWaitEvent(dummy_unifying_free_stream.stream(), event));
      C10_CUDA_CHECK(cudaEventDestroy(event));
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

    if (C10_UNLIKELY(captures_underway)) {
      // See Note [Avoid dangling free streams during CUDA graph capture]
      capture_free_streams.insert(dummy_unifying_free_stream);
    }
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

  std::lock_guard<std::mutex> lk(general_mutex);

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
  std::lock_guard<std::mutex> lk(general_mutex);

  for (int dev = 0; dev < device_count; dev++) {
    if (devs_initialized_flags[dev]) {
      CUDAGuard g(dev);

      cudaMemPool_t mempool;
      cudaDeviceGetDefaultMemPool(&mempool, dev);
      cudaDeviceSynchronize();
      cudaMemPoolTrimTo(mempool, 0);
    }
  }
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
}

void* getBaseAllocation(void* ptr, size_t* size) {
  if (size) {
    // maybe we do need to track per-ptr sizes after all
  }
  return ptr;
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
  std::lock_guard<std::mutex> lk(general_mutex);

  // The pointer should exist in the map already.
  auto it = usage_streams_each_ptr.find(ptr.get());
  TORCH_INTERNAL_ASSERT(it != usage_streams_each_ptr.end(),
                        "ptr not represented in usage_streams_each_ptr");
  TORCH_INTERNAL_ASSERT(it->second.size() != 0,
                        "ptr's stream uses vector is empty");

  it->second.push_back(stream);
}

std::mutex* getFreeMutex() {
  return &general_mutex;
}

static inline void assertValidDevice(int device) {
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

  std::lock_guard<std::mutex> lk(general_mutex);

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

  // just to get it to compile, WIP
  return {};
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
                                         &zero));
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(mempool,
                                         cudaMemPoolAttrUsedMemHigh,
                                         &zero));
}

std::vector<SegmentInfo> snapshot() {
  return {};
}

// CUDAGraph interactions
void notifyCaptureBegin(CaptureId_t graph_id, MempoolId_t mempool_id) {} // no-op

void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) {
  assertValidDevice(device);

  auto capture_stream = cuda::getCurrentCUDAStream(device)

  // See Note [Avoid dangling free streams during CUDA graph capture]
  for (const auto& free_stream : capture_free_streams) {
    // cudaEventRecord requires that the input event and stream are on the same device.
    CUDAGuard g(free_stream.device_index());

    // CUDACachingAllocator.cpp uses raw cuda events, as do we.
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(event, free_stream.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(capture_stream.stream(), event));
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }
}

void notifyCaptureEnded(int device, CaptureId_t graph_id) {
  assertValidDevice(device);

}

void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {} // no-op

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
