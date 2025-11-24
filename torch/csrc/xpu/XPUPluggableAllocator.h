#pragma once

#include <c10/xpu/XPUCachingAllocator.h>

namespace torch::xpu::XPUPluggableAllocator {

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      sycl::queue* queue)
      : size(size), device_idx(device_idx), queue(queue) {};
  size_t size{0};
  c10::DeviceIndex device_idx{-1};
  sycl::queue* queue{};
};

struct TORCH_XPU_API XPUPluggableAllocator
    : public c10::xpu::XPUCachingAllocator::XPUAllocator {
  XPUPluggableAllocator(
      std::function<void*(size_t, int, sycl::queue*)> alloc_fn,
      std::function<void(void*, size_t, int, sycl::queue*)> free_fn);

  XPUPluggableAllocator(XPUPluggableAllocator& other) = default;
  XPUPluggableAllocator(XPUPluggableAllocator&& other) = delete;
  XPUPluggableAllocator& operator=(const XPUPluggableAllocator& other) = delete;
  XPUPluggableAllocator& operator=(XPUPluggableAllocator&& other) = delete;

  ~XPUPluggableAllocator() override = default;

  void* malloc(size_t size, c10::DeviceIndex device, sycl::queue stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void raw_delete(void* ptr) override;
  bool initialized() override;

 protected:
  std::function<void*(size_t, int, sycl::queue*)> alloc_fn_;
  std::function<void(void*, size_t, int, sycl::queue*)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void(void* ptr, sycl::queue*)> record_stream_fn_;
  std::mutex allocator_mutex_;
  // We do the bookkeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;
};

} // namespace torch::xpu::XPUPluggableAllocator
