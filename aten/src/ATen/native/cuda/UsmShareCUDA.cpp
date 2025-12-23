#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Allocator.h>

namespace at::native {

// Implementation for CUDA backend
// self: dummy tensor on CUDA device (carries device info)
// src:  CPU storage containing the data
Tensor usm_share_from_cuda(const Tensor& self, c10::Storage src) {
  void* src_ptr = src.data_ptr().get();
  size_t src_bytes = src.nbytes();
  c10::Device target_device = self.device();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_cuda: source storage must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      target_device.type() == c10::DeviceType::CUDA,
      "usm_share_from_cuda: target device must be CUDA, got: ",
      target_device);

  // Handle empty case
  if (src_bytes == 0) {
    return at::empty_like(self, self.options());
  }

  // Set device guard
  c10::cuda::CUDAGuard device_guard(target_device);
  c10::Device actual_device = device_guard.current_device();

  // Check device USM support (integrated GPU only)
  auto device_properties = at::cuda::getDeviceProperties(actual_device.index());
  TORCH_CHECK(
      device_properties->integrated,
      "usm_share_from_cuda: target device does not support USM (requires integrated GPU): ",
      actual_device);

  // Register host memory for CUDA access
  cudaError_t err = cudaHostRegister(src_ptr, src_bytes, cudaHostRegisterDefault);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_from_cuda: cudaHostRegister failed, error: ",
      cudaGetErrorString(err));

  // Get device pointer
  void* dev_ptr = src_ptr;
  err = cudaHostGetDevicePointer(&dev_ptr, src_ptr, 0);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_from_cuda: cudaHostGetDevicePointer failed, error: ",
      cudaGetErrorString(err));

  // Create a new storage with a custom deleter that also updates src metadata
  c10::StorageImpl* src_impl = src.unsafeGetStorageImpl();
  // Increment ref count of src to prevent it from being freed while the new
  // storage shares its data. The ref count will be decremented in the deleter.
  c10::raw::intrusive_ptr::incref(src_impl);

  struct DeleterContext {
    c10::StorageImpl* src_impl{};
    void* host_ptr{};  // Original CPU pointer for unregister
    void* device_ptr{};  // Device pointer (not used in deleter, just for reference)
    c10::Device device;
  };

  auto* deleter_context = new DeleterContext{src_impl, src_ptr, dev_ptr, actual_device};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    // Set device guard
    c10::cuda::CUDAGuard device_guard(context->device);
    // Unregister the host memory using the original host pointer
    cudaError_t err = cudaHostUnregister(context->host_ptr);
    if (err != cudaSuccess) {
      TORCH_WARN(
          "usm_share_cuda: cudaHostUnregister failed on device ",
          context->device,
          " with error: ",
          cudaGetErrorString(err));
    }
    // Decrement the ref count of src
    c10::raw::intrusive_ptr::decref(context->src_impl);
    delete context;
  };

  auto data_ptr = c10::DataPtr(dev_ptr, deleter_context, deleter, actual_device);

  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      src_bytes,
      std::move(data_ptr),
      c10::GetAllocator(c10::DeviceType::CUDA),
      /* resizable */ false);

  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
