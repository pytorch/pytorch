#pragma once

#include <c10/xpu/XPUCachingAllocator.h>

namespace torch::xpu::XPUPluggableAllocator {

struct TORCH_XPU_API XPUPluggableAllocator
    : public c10::xpu::XPUCachingAllocator::XPUAllocator {
  XPUPluggableAllocator(
      std::function<void*(size_t, int, sycl::queue)> alloc_fn,
      std::function<void(void*, size_t, int, sycl::queue)> free_fn);

  XPUPluggableAllocator(XPUPluggableAllocator& other);
  XPUPluggableAllocator(XPUPluggableAllocator&& other) = delete;
  XPUPluggableAllocator& operator=(const XPUPluggableAllocator& other) = delete;
  XPUPluggableAllocator& operator=(XPUPluggableAllocator&& other) = delete;

  ~XPUPluggableAllocator() override = default;

  void* malloc(size_t size, c10::DeviceIndex device, cudaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;

 protected:
  std::function<void*(size_t, int, sycl::queue)> alloc_fn_;
  std::function<void(void*, size_t, int, sycl::queue)> free_fn_;
  bool initialized_ = false;
};

} // namespace torch::xpu::XPUPluggableAllocator
