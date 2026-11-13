#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

struct ShareableHandle {
  ptrdiff_t offset;
  std::string handle;
};

class XPUAllocator : public DeviceAllocator {
 public:
  virtual void init(c10::DeviceIndex device_count) = 0;
  virtual void* raw_alloc(size_t size) = 0;
  virtual void raw_delete(void* ptr) = 0;
  std::pair<size_t, size_t> getMemoryInfo(
      c10::DeviceIndex device_index) override {
    const auto& device = c10::xpu::get_raw_device(device_index);
    const size_t total = device.get_info<sycl::info::device::global_mem_size>();
    TORCH_CHECK(
        device.has(sycl::aspect::ext_intel_free_memory),
        "The device (",
        device.get_info<sycl::info::device::name>(),
        ") doesn't support querying the available free memory. ",
        "You can file an issue at https://github.com/pytorch/pytorch/issues ",
        "to help us prioritize its implementation.");
    const size_t free =
        device.get_info<sycl::ext::intel::info::device::free_memory>();

#if SYCL_COMPILER_VERSION >= 20260200
    namespace syclex = sycl::ext::oneapi::experimental;
    const auto arch = device.get_info<syclex::info::device::architecture>();
    if (arch < syclex::architecture::intel_gpu_bmg_g21) {
      return {free, total};
    }
    // See
    // https://github.com/intel/compute-runtime/blob/master/programmers-guide/DEVICE_MEMORY_ACCOUNTING.md#umd-headroom.
    constexpr double kIntegratedGpuUsableFraction = 0.94;
#ifdef _WIN32
    constexpr double kDiscreteGpuUsableFraction = 0.98;
#else
    constexpr double kDiscreteGpuUsableFraction = 0.95;
#endif
    const double usable_fraction =
        device.has(sycl::aspect::ext_oneapi_is_integrated_gpu)
        ? kIntegratedGpuUsableFraction
        : kDiscreteGpuUsableFraction;
    const size_t free_adjust =
        free + static_cast<size_t>((1 - usable_fraction) * total);
    TORCH_CHECK(
        free_adjust <= total, "Calculated free memory exceeds total memory.");
    return {free_adjust, total};
#else
    return {free, total};
#endif
  }
};

C10_XPU_API extern std::atomic<XPUAllocator*> allocator;

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<c10::DataPtr> dataptrs_allocd;
};

struct AllocatorConfigInfo {
  bool expandable_segments;
  std::string last_allocator_settings;
};

struct SnapshotInfo {
  std::vector<CachingDeviceAllocator::SegmentInfo> segments;
  std::vector<std::vector<CachingDeviceAllocator::TraceEntry>> device_traces;
  AllocatorConfigInfo config_metadata;
};

inline XPUAllocator* get() {
  return allocator.load();
}

inline void init(c10::DeviceIndex device_count) {
  get()->init(device_count);
}

inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  get()->emptyCache(mempool_id);
}

inline void resetPeakStats(DeviceIndex device) {
  get()->resetPeakStats(device);
}

inline void resetAccumulatedStats(DeviceIndex device) {
  get()->resetAccumulatedStats(device);
}

inline c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void* raw_alloc(size_t size) {
  return get()->raw_alloc(size);
}

inline void raw_delete(void* ptr) {
  get()->raw_delete(ptr);
}

inline void recordStream(const DataPtr& dataPtr, XPUStream stream) {
  get()->recordStream(dataPtr, stream);
}

C10_XPU_API void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

C10_XPU_API double getMemoryFraction(DeviceIndex device);

C10_XPU_API void setMemoryFraction(double fraction, DeviceIndex device);

C10_XPU_API void recordHistory(
    bool enabled,
    CachingDeviceAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    CachingDeviceAllocator::RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions);

C10_XPU_API void attachAllocatorTraceTracker(
    CachingDeviceAllocator::AllocatorTraceTracker tracker);

C10_XPU_API SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0});

C10_XPU_API ShareableHandle shareIpcHandle(void* ptr);

C10_XPU_API std::shared_ptr<void> getIpcDevPtr(std::string handle);

C10_XPU_API void createOrIncrefPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id,
    std::shared_ptr<XPUAllocator> allocator_ptr = nullptr);

C10_XPU_API void beginAllocateToPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id,
    std::function<bool(sycl::queue*)> filter);

C10_XPU_API void endAllocateToPool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

C10_XPU_API void markCaptureBegin(c10::DeviceIndex device);

C10_XPU_API void markCaptureEnd(c10::DeviceIndex device);

C10_XPU_API void releasePool(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

C10_XPU_API void setNoSplit(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

// Register/unregister a pool as an OOM fallback. Callers must explicitly
// call setUseOnOOM(..., false) before releasing the pool.
C10_XPU_API void setUseOnOOM(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id,
    bool use_on_oom);

C10_XPU_API int getPoolUseCount(
    c10::DeviceIndex device,
    c10::MempoolId_t mempool_id);

C10_XPU_API std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id);

C10_XPU_API CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> as);

C10_XPU_API bool isHistoryEnabled();

C10_XPU_API bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations);

} // namespace c10::xpu::XPUCachingAllocator

namespace c10::xpu {
// Keep BC only
using c10::CaptureId_t;
using c10::MempoolId_t;
} // namespace c10::xpu
