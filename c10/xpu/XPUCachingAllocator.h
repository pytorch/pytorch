#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/ApproximateClock.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

typedef std::shared_ptr<GatheredContext> (*CreateContextFn)();

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext> context_when_allocated;
};

// Struct containing info of a memory segment (i.e. one contiguous device memory
// allocation).
struct SegmentInfo {
  c10::DeviceIndex device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0;
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
    ALLOC,
    FREE_REQUESTED,
    FREE_COMPLETED,
    SEGMENT_ALLOC,
    SEGMENT_FREE,
    SEGMENT_MAP,
    SEGMENT_UNMAP,
    SNAPSHOT,
    OOM
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
  size_t addr_;
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

C10_XPU_API void emptyCache(MempoolId_t mempool_id = {0, 0});

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

C10_XPU_API void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when,
    bool clearHistory);

class XPUAllocator;

C10_XPU_API void createOrIncrefPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id,
    XPUAllocator* allocator = nullptr);

C10_XPU_API void beginAllocateToPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id,
    std::function<bool(sycl::queue*)> filter);

C10_XPU_API void endAllocateToPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

C10_XPU_API void releasePool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

C10_XPU_API int getPoolUseCount(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

} // namespace c10::xpu::XPUCachingAllocator

namespace c10::xpu {

using c10::CaptureId_t;
using c10::MempoolId_t;
struct C10_XPU_API MemPool {
  MemPool(
      XPUCachingAllocator::XPUAllocator* allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  XPUCachingAllocator::XPUAllocator* allocator();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  XPUCachingAllocator::XPUAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};
} // namespace c10::xpu
