#include <torch/csrc/xpu/XPUPluggableAllocator.h>

namespace torch::xpu::XPUPluggableAllocator {

void custom_raw_deleter(void* ptr);

static c10::DeviceIndex device_count_ = 0;

void* XPUPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    sycl::queue* queue) {
  void* r = alloc_fn_(size, device, queue);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, queue));
  }
  return r;
}

c10::DataPtr XPUPluggableAllocator::allocate(size_t size) {
  auto device = c10::xpu::current_device();
  sycl::queue& queue = c10::xpu::getCurrentXPUStream(device);
  void* r = this->malloc(size, device, &queue);
  return {r, r, raw_deleter(), c10::Device(c10::kXPU, device)};
}

void* XPUPluggableAllocator::raw_alloc(size_t nbytes) {
  auto device = c10::xpu::current_device();
  sycl::queue& queue = c10::xpu::getCurrentXPUStream(device);
  return malloc(nbytes, device, &queue);
}

c10::DeleterFnPtr XPUPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void XPUPluggableAllocator::raw_delete(void* ptr) {
  sycl::queue* queue = nullptr;
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    queue = metadata.queue;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, queue);
}

void XPUPluggableAllocator::init(c10::DeviceIndex device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  device_count_ = device_count;
  initialized_ = true;
}

bool XPUPluggableAllocator::initialized() {
  return initialized_;
}

void XPUPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  c10::xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
}

void XPUPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    c10::Stream stream) {
  if (record_stream_fn_) {
    auto xpu_stream = c10::xpu::XPUStream(stream);
    record_stream_fn_(ptr.get(), &xpu_stream.queue());
  }
}

void XPUPluggableAllocator::emptyCache(
    /*unused*/ c10::MempoolId_t mempool_id) {
  TORCH_CHECK(
      false,
      "XPUPluggableAllocator does not yet support emptyCache. "
      "If you need it, please file an issue describing your use case.");
}

c10::CachingDeviceAllocator::DeviceStats XPUPluggableAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "XPUPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void XPUPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "XPUPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void XPUPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "XPUPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
    current_custom_allocator;

std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, sycl::queue*)> alloc_fn,
    std::function<void(void*, size_t, int, sycl::queue*)> free_fn) {
  auto allocator = std::make_shared<XPUPluggableAllocator>(
      std::move(alloc_fn), std::move(free_fn));
  allocator->init(device_count_);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::xpu::XPUCachingAllocator::get()->initialized(),
      "Can't swap an already initialized allocator");
  c10::xpu::XPUCachingAllocator::allocator.store(allocator.get());
  c10::SetAllocator(c10::kXPU, allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

} // namespace torch::xpu::XPUPluggableAllocator
