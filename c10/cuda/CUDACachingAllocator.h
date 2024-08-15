#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

namespace c10 {

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
} // namespace c10
  //
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

namespace c10::cuda::CUDACachingAllocator {

extern const size_t kLargeBuffer;

typedef std::shared_ptr<GatheredContext> (*CreateContextFn)();

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated; // per-watcher context
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  c10::DeviceIndex device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0; // unrounded, actually requested size
  size_t allocated_size = 0;
  size_t active_size = 0;
  cudaStream_t stream = nullptr;
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0};
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
};

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

union trace_time_ {
  time_t t_;
  approx_time_t approx_t_;
};

struct TraceEntry {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via record_stream
                    // This event is generated when a free actually completes.
    SEGMENT_ALLOC, // a call to cudaMalloc to get more memory from the OS
    SEGMENT_FREE, // a call to cudaFree to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // a call to cuMemMap (used with expandable_segments)
    SEGMENT_UNMAP, // unmap part of a segment (used with expandable segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by cuda)
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      size_t addr,
      size_t size,
      cudaStream_t stream,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr)
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size) {
    time_.approx_t_ = time;
  }
  Action action_;
  c10::DeviceIndex device_;
  size_t addr_; // for OOM, this is the amount of free bytes reported by cuda
  std::shared_ptr<GatheredContext> context_;
  cudaStream_t stream_{};
  size_t size_;
  trace_time_ time_{};
};

// Calls made by record_function will save annotations
struct AnnotationEntry {
  AnnotationEntry(c10::DeviceIndex device, approx_time_t time)
      : device_(device) {
    time_.approx_t_ = time;
  }

  void recordUserMetadata(const std::string& name, std::string value) {
    metadata_[name] = std::move(value);
  }

  c10::DeviceIndex device_;
  trace_time_ time_{};
  std::unordered_map<std::string, std::string> metadata_;
};

struct AllocatorConfigInfo {
  double garbage_collection_threshold;
  size_t max_split_size;
  size_t pinned_num_register_threads;
  bool expandable_segments;
  bool release_lock_on_malloc;
  bool pinned_use_host_register;
  std::string last_allocator_settings;
  std::vector<size_t> roundup_power2_divisions;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
  std::vector<AnnotationEntry> external_annotations;
  AllocatorConfigInfo config_metadata;
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<at::DataPtr> dataptrs_allocd;
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

// Size pretty-printer
std::string format_size(uint64_t size);

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    size_t allocated,
    size_t device_total,
    size_t device_free)>;

using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

struct ShareableHandle {
  ptrdiff_t offset;
  std::string handle;
};

class CUDAAllocator : public Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual bool initialized() = 0;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
  virtual void emptyCache() = 0;
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const DataPtr&, CUDAStream stream) = 0;
  virtual DeviceStats getDeviceStats(c10::DeviceIndex device) = 0;
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) = 0;
  virtual void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) = 0;
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  // returns true if the allocated blocks are equal to expected live allocations
  virtual bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual ShareableHandle shareIpcHandle(void* ptr) = 0;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) = 0;
  virtual void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& md){};
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // Attached AllocatorTraceTracker callbacks will be called while the
  // per-device allocator lock is held. Any additional locks taken from within
  // the callback must be proven to always have the lock order that never
  // triggers a deadlock. In particular, Python's GIL may be held when
  // calling the allocator so it is unsafe to try to acquire the GIL in this
  // callback.
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  virtual void enablePeerAccess(
      c10::DeviceIndex dev,
      c10::DeviceIndex dev_to_access) = 0;

  // memory not allocated from cudaMalloc cannot be copied
  // across devices using cudaMemcpyAsync if peer to peer access is disabled.
  // instead it requires cudaMemcpyAsyncPeer
  //  with P2P Enabled, all combinations work
  //  with P2P Disabled:
  //                       cudaMalloc cudaMallocAsync/cuMemMap
  // cudaMemcpyAsyncPeer   works      works
  // cudaMemcpyAsync       works      error

  // This function performs chooses to use the Peer version of
  // memcpy if required based on where the allocated put dst/src.
  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) = 0;
  virtual std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) = 0;
  virtual CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) = 0;
  virtual std::string name() = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in CUDACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
C10_CUDA_API extern std::atomic<CUDAAllocator*> allocator;

inline CUDAAllocator* get() {
  return allocator.load();
}

// Called directly by clients.
inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void init(int device_count) {
  return get()->init(device_count);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

inline void emptyCache() {
  return get()->emptyCache();
}

inline void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  return get()->cacheInfo(device, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const DataPtr& dataPtr, CUDAStream stream) {
  return get()->recordStream(dataPtr, stream);
}

inline DeviceStats getDeviceStats(c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot() {
  return get()->snapshot();
}

inline std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

// CUDAGraph interactions
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) {
  return get()->recordHistory(
      enabled, context_recorder, alloc_trace_max_entries, when);
}

inline void recordAnnotation(
    const std::vector<std::pair<std::string, std::string>>& md) {
  return get()->recordAnnotation(md);
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(
      device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(std::move(observer));
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}
// Not part of CUDA_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

inline ShareableHandle shareIpcHandle(void* ptr) {
  return get()->shareIpcHandle(ptr);
}

inline std::string name() {
  return get()->name();
}

inline cudaError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(
      dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

inline void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

} // namespace c10::cuda::CUDACachingAllocator

namespace c10::cuda {

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the CUDACachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct C10_CUDA_API MemPool {
  MemPool(
      CUDACachingAllocator::CUDAAllocator* allocator = nullptr,
      bool is_user_created = true);

  MempoolId_t id();
  CUDACachingAllocator::CUDAAllocator* allocator();

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  CUDACachingAllocator::CUDAAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
};

// MemPoolContext holds the currently active pool and stashes the previous
// pool. On deletion it makes the previous pool active.
struct C10_CUDA_API MemPoolContext {
  MemPoolContext(MemPool* mempool);

  ~MemPoolContext();

  // getActiveMemPool() can be used to get the currently active pool.
  // For instance: in CUDACachingAllocator, we can route allocations
  // to a user provided allocator, by doing:
  //
  //  auto active_pool = MemPoolContext::getActiveMemPool();
  //  if (active_pool && active_pool->allocator()) {
  //    ptr = active_pool->allocator()->raw_alloc(size);
  //  }
  //
  static MemPool* getActiveMemPool();

 private:
  MemPool* prev_mempool_;
};

} // namespace c10::cuda
