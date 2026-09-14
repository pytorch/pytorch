#pragma once

#include <c10/core/Allocator.h>
#include <c10/xpu/XPUCachingAllocator.h>

namespace at::xpu {

// Keep BC only
using c10::CaptureId_t;
using c10::MempoolId_t;

struct TORCH_XPU_API MemPool {
  MemPool(
      XPUCachingAllocator::XPUAllocator* allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false);

  C10_DISABLE_COPY_AND_ASSIGN(MemPool);
  MemPool(MemPool&&) = default;
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

} // namespace at::xpu
