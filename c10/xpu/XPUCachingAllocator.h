#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/ApproximateClock.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

// Struct containing info of an allocation block
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated; // per-watcher context
};

// Struct containing info of a memory segment (i.e. one contiguous device memory
// allocation).
struct SegmentInfo {
  c10::DeviceIndex device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0; // unrounded, actually requested size
  size_t allocated_size = 0;
  size_t active_size = 0;
  sycl::queue* queue = nullptr;
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
    SEGMENT_ALLOC, // a call to SYCL runtime to get more memory from the OS
    SEGMENT_FREE, // a call to SYCL runtime to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // a call to map virtual memory (used with expandable_segments)
    SEGMENT_UNMAP, // a call to unmap part of a segment (used with expandable
                   // segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by SYCL runtime)
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      size_t addr,
      size_t size,
      sycl::queue* queue,
      MempoolId_t mempool,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr,
      std::string compile_context = "",
      std::string user_metadata = "")
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        queue_(queue),
        size_(size),
        mempool_(std::move(mempool)),
        compile_context_(std::move(compile_context)),
        user_metadata_(std::move(user_metadata)) {
    time_.approx_t_ = time;
  }
  Action action_;
  c10::DeviceIndex device_;
  size_t addr_; // for OOM, this is the amount of free bytes reported by runtime
  std::shared_ptr<GatheredContext> context_;
  sycl::queue* queue_{};
  size_t size_;
  MempoolId_t mempool_;
  trace_time_ time_{};
  std::string compile_context_;
  std::string user_metadata_;
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
  bool expandable_segments;
  std::vector<size_t> roundup_power2_divisions;
  std::string last_allocator_settings;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
  std::vector<AnnotationEntry> external_annotations;
  AllocatorConfigInfo config_metadata;
};

C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void resetPeakStats(DeviceIndex device);

C10_XPU_API void resetAccumulatedStats(DeviceIndex device);

C10_XPU_API c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device);

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

C10_XPU_API void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

C10_XPU_API double getMemoryFraction(DeviceIndex device);

C10_XPU_API void setMemoryFraction(double fraction, DeviceIndex device);

C10_XPU_API SnapshotInfo snapshot();

} // namespace c10::xpu::XPUCachingAllocator
