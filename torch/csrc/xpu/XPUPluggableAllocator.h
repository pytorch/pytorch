#pragma once

#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/Export.h>

namespace torch::xpu::XPUPluggableAllocator {

struct _AllocationMetadata {
  _AllocationMetadata() {}
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      sycl::queue* queue)
      : size(size), device_idx(device_idx), queue(queue) {}
  size_t size{0};
  c10::DeviceIndex device_idx{-1};
  sycl::queue* queue{};
};

struct TORCH_PYTHON_API XPUPluggableAllocator
    : public c10::xpu::XPUCachingAllocator::XPUAllocator {
  XPUPluggableAllocator(
      std::function<void*(size_t, int, sycl::queue*)> alloc_fn,
      std::function<void(void*, size_t, int, sycl::queue*)> free_fn)
      : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

  C10_DISABLE_COPY_AND_ASSIGN(XPUPluggableAllocator);

  ~XPUPluggableAllocator() override = default;

  void* malloc(size_t size, c10::DeviceIndex device, sycl::queue* stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void raw_delete(void* ptr) override;
  void init(c10::DeviceIndex device_count) override;
  bool initialized() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

  void recordStream(const c10::DataPtr&, c10::Stream stream) override;
  void emptyCache(c10::MempoolId_t mempool_id = {0, 0}) override;
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;

  void set_init_fn(std::function<void(int)> init_fn) {
    init_fn_ = std::move(init_fn);
  }
  void set_record_stream_fn(
      std::function<void(void* ptr, sycl::queue* queue)> record_stream_fn) {
    record_stream_fn_ = std::move(record_stream_fn);
  }

 protected:
  std::function<void*(size_t, int, sycl::queue*)> alloc_fn_;
  std::function<void(void*, size_t, int, sycl::queue*)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void(void* ptr, sycl::queue*)> record_stream_fn_;
  std::mutex allocator_mutex_;
  // We do the bookkeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;
  bool initialized_ = false;
};

TORCH_XPU_API std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
getCurrentAllocator();

TORCH_XPU_API std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, sycl::queue*)> alloc_fn,
    std::function<void(void*, size_t, int, sycl::queue*)> free_fn);

TORCH_XPU_API void changeCurrentAllocator(
    const std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>&
        allocator);

} // namespace torch::xpu::XPUPluggableAllocator
