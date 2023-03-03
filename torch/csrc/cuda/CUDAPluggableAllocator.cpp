#include <mutex>
#include <unordered_map>
#include <utility>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace torch {
namespace cuda {
namespace CUDAPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream(0) {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    int device_idx,
    cudaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
CUDAPluggableAllocator::CUDAPluggableAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
    : alloc_fn_(alloc_fn), free_fn_(free_fn) {}

CUDAPluggableAllocator::CUDAPluggableAllocator(CUDAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      capture_begin_fn_(other.capture_begin_fn_),
      capture_about_to_end_fn_(other.capture_about_to_end_fn_),
      capture_ended_fn_(other.capture_ended_fn_),
      capture_destroy_fn_(other.capture_destroy_fn_) {}

void CUDAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = init_fn;
}

void CUDAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = reset_fn;
}

void CUDAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = memory_fraction_fn;
}

void CUDAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = base_alloc_fn;
}

void CUDAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = record_stream_fn;
}

void CUDAPluggableAllocator::set_capture_begin_fn(
    std::function<void(int, c10::cuda::CaptureId_t, c10::cuda::MempoolId_t)>
        capture_begin_fn) {
  capture_begin_fn_ = capture_begin_fn;
}

void CUDAPluggableAllocator::set_capture_about_to_end_fn(
    std::function<void(int, c10::cuda::CaptureId_t)> capture_about_to_end_fn) {
  capture_about_to_end_fn_ = capture_about_to_end_fn;
}

void CUDAPluggableAllocator::set_capture_ended_fn(
    std::function<void(int, c10::cuda::CaptureId_t)> capture_ended_fn) {
  capture_ended_fn_ = capture_ended_fn;
}

void CUDAPluggableAllocator::set_capture_destroy_fn(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn) {
  capture_destroy_fn_ = capture_destroy_fn;
}

void* CUDAPluggableAllocator::malloc(
    size_t size,
    int device,
    cudaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr CUDAPluggableAllocator::allocate(size_t size) const {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  void* r =
      const_cast<CUDAPluggableAllocator*>(this)->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  return data_ptr;
}

c10::DeleterFnPtr CUDAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* CUDAPluggableAllocator::raw_alloc(size_t nbytes) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  return malloc(nbytes, device, stream);
}

void* CUDAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  return malloc(nbytes, device, stream);
}

void CUDAPluggableAllocator::raw_delete(void* ptr) {
  cudaStream_t stream;
  int device_idx;
  size_t size;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void CUDAPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool CUDAPluggableAllocator::initialized() {
  return initialized_;
}

void CUDAPluggableAllocator::setMemoryFraction(double fraction, int device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void CUDAPluggableAllocator::emptyCache(void) {
  if (reset_fn_) {
    return reset_fn_();
  }
}

void CUDAPluggableAllocator::cacheInfo(int dev_id, size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* CUDAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void CUDAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::cuda::CUDACachingAllocator::DeviceStats CUDAPluggableAllocator::
    getDeviceStats(int device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetAccumulatedStats(int device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetPeakStats(int device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAPluggableAllocator::
    snapshot() {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> CUDAPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void CUDAPluggableAllocator::notifyCaptureBegin(
    int device,
    c10::cuda::CaptureId_t graph_id,
    c10::cuda::MempoolId_t mempool_id) {
  if (capture_begin_fn_) {
    capture_begin_fn_(device, graph_id, mempool_id);
  }
}

void CUDAPluggableAllocator::notifyCaptureAboutToEnd(
    int device,
    c10::cuda::CaptureId_t graph_id) {
  if (capture_about_to_end_fn_) {
    capture_about_to_end_fn_(device, graph_id);
  }
}

void CUDAPluggableAllocator::notifyCaptureEnded(
    int device,
    c10::cuda::CaptureId_t graph_id) {
  if (capture_ended_fn_) {
    capture_ended_fn_(device, graph_id);
  }
}

void CUDAPluggableAllocator::notifyCaptureDestroy(
    int device,
    c10::cuda::MempoolId_t mempool_id) {
  if (capture_destroy_fn_) {
    capture_destroy_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    bool alloc_trace_record_context) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

bool CUDAPluggableAllocator::needsPoolSpecificPeerAccess() {
  return false;
}

std::string CUDAPluggableAllocator::name() {
  return "pluggable";
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
  std::shared_ptr<CUDAPluggableAllocator> allocator(
      new CUDAPluggableAllocator(alloc_fn, free_fn));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) {
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

} // namespace CUDAPluggableAllocator
} // namespace cuda
} // namespace torch
