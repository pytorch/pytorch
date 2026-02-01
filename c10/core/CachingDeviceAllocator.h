#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/util/ApproximateClock.h>

namespace c10::CachingDeviceAllocator {

using namespace c10::CachingAllocator;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory allocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to device malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to device memory allocation
  // after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

using CreateContextFn = std::shared_ptr<GatheredContext> (*)();

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

// Struct containing information about an allocation block, i.e., a subrange
// of a device allocation (such as one obtained via cudaMalloc).
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated; // per-watcher context
};

// Struct holding information about a memory segment (i.e., a single contiguous
// device allocation, such as one created by cudaMalloc).
struct SegmentInfo {
  c10::DeviceIndex device = 0;
  int32_t registration_counter = -1;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0; // Unrounded, actually requested size
  size_t allocated_size = 0;
  size_t active_size = 0;
  void* stream = nullptr; // Records the address of the underlying stream
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0};
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
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
    SEGMENT_ALLOC, // a call to device allocation to get more memory from the OS
    SEGMENT_FREE, // a call to device deallocation to return memory to the OS
                  // (e.g. to defragment or empty_caches)
    SEGMENT_MAP, // a call to cuMemMap (used with expandable_segments)
    SEGMENT_UNMAP, // unmap part of a segment (used with expandable segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by device memory)
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      size_t addr,
      size_t size,
      void* stream,
      MempoolId_t mempool,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr,
      std::string compile_context = "",
      std::string user_metadata = "")
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size),
        mempool_(std::move(mempool)),
        compile_context_(std::move(compile_context)),
        user_metadata_(std::move(user_metadata)) {
    time_.approx_t_ = time;
  }
  Action action_;
  c10::DeviceIndex device_;
  // For most actions, this is a memory address. For OOM, it represents the
  // amount of free memory (in bytes). For SNAPSHOT, it is an unused parameter
  // (just set to 0).
  size_t addr_;
  std::shared_ptr<GatheredContext> context_;
  void* stream_{};
  size_t size_;
  MempoolId_t mempool_;
  trace_time_ time_{};
  std::string compile_context_;
  std::string user_metadata_;
};

inline TraceEntry::Action parseTraceEntryAction(std::string_view action) {
  constexpr std::pair<std::string_view, TraceEntry::Action> kActionTable[] = {
      {"alloc", TraceEntry::Action::ALLOC},
      {"free_requested", TraceEntry::Action::FREE_REQUESTED},
      {"free_completed", TraceEntry::Action::FREE_COMPLETED},
      {"segment_alloc", TraceEntry::Action::SEGMENT_ALLOC},
      {"segment_free", TraceEntry::Action::SEGMENT_FREE},
      {"segment_map", TraceEntry::Action::SEGMENT_MAP},
      {"segment_unmap", TraceEntry::Action::SEGMENT_UNMAP},
      {"snapshot", TraceEntry::Action::SNAPSHOT},
      {"oom", TraceEntry::Action::OOM},
  };
  for (const auto& [k, v] : kActionTable) {
    if (action == k)
      return v;
  }
  TORCH_CHECK(false, "Unknown TraceEntry action: ", action);
}

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

using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

} // namespace c10::CachingDeviceAllocator

namespace c10 {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by Graph mode capture_begin.
// second is set if the instance is created by Graph mode graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

struct C10_API DeviceAllocator : public c10::Allocator {
  DeviceAllocator();
  ~DeviceAllocator() override;

  // Returns true if the allocator has been properly initialized and is ready
  // for use
  virtual bool initialized() = 0;

  // Releases all cached device memory from the specified memory pool back to
  // the system
  virtual void emptyCache(MempoolId_t mempool_id = {0, 0}) = 0;

  // Associates a memory allocation with a stream to establish dependency
  // tracking. Prevents memory reuse until all operations on the specified
  // stream complete
  virtual void recordStream(const DataPtr& ptr, c10::Stream stream) = 0;

  // Retrieves comprehensive memory statistics for the specified device,
  // including allocation patterns, usage metrics
  virtual CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) = 0;

  // Resets cumulative allocation statistics for the specified device to zero
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;

  // Resets peak memory usage statistics for the specified device
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;

  // Return the free memory size and total memory size in bytes for the
  // specified device.
  virtual std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "getMemoryInfo is not implemented for this allocator yet.");
  }
};

// This function is used to get the DeviceAllocator for a specific device type
// and keep backward compatibility with c10::GetAllocator.
C10_API inline DeviceAllocator* getDeviceAllocator(const DeviceType& t) {
  TORCH_CHECK(
      t != DeviceType::CPU,
      "getDeviceAllocator is not supported for CPU device type.");
  auto* allocator = c10::GetAllocator(t);
  auto* device_allocator = dynamic_cast<DeviceAllocator*>(allocator);
  TORCH_INTERNAL_ASSERT(
      device_allocator, "Allocator for ", t, " is not a DeviceAllocator.");
  return device_allocator;
}

} // namespace c10
