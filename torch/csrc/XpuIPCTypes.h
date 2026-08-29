#pragma once

#ifdef USE_XPU

#include <ATen/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace torch {

struct XpuSharedStorage final {
  c10::DeviceIndex device{-1};
  std::string handle;
  std::string ref_counter_handle;
  uint64_t ref_counter_offset{0};
  size_t size_bytes{0};
  ptrdiff_t offset_bytes{0};
};

C10_XPU_API XpuSharedStorage ShareXpuStorage(const at::Storage& storage);

C10_XPU_API c10::intrusive_ptr<at::StorageImpl> NewStorageFromXpuShared(
    const XpuSharedStorage& shared);

C10_XPU_API void ReleaseXpuIPCRefCounter(
  const std::string& handle,
  uint64_t offset);

C10_XPU_API bool XpuIPCCollect();

C10_XPU_API bool IsImportedXpuStorage(const c10::StorageImpl& storage);

} // namespace torch

namespace c10::xpu::XPUCachingAllocator {

class XpuIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch::XpuIPCCollect();
  }
};

} // namespace c10::xpu::XPUCachingAllocator

#endif
