#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

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
