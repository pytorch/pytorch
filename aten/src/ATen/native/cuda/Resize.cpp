#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <torch/library.h>
#include <ATen/native/cuda/Resize.h>

namespace at {
namespace native {

void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  auto device = at::cuda::current_device();
  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(at::DeviceType::CUDA, device)));
    storage->set_nbytes(0);
    return;
  }

  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    // Enable p2p access when the memcpy is across devices
    at::globalContext().lazyInitCUDA();
    at::cuda::get_p2p_access(device, storage->device().index());

    C10_CUDA_CHECK(
        cudaMemcpyAsync(
            data.get(),
            storage->data(),
            std::min(storage->nbytes(), size_bytes),
            cudaMemcpyDeviceToDevice,
            c10::cuda::getCurrentCUDAStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard) {
  cuda::OptionalCUDAGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }
  resize_impl_template_<&maybe_resize_storage_cuda, &select_storage_size_default>(
      self, size, stride);
  return self;
}

const Tensor& resize_cuda_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  resize_template<&resize_impl_cuda_>(self, size, c10::nullopt, optional_memory_format, true);
  return self;
}

const Tensor& resize_with_strides_cuda_(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef strides) {
  cuda::OptionalCUDAGuard guard(self.storage().device().index());
  resize_template<&resize_impl_tryreuse_<&maybe_resize_storage_cuda>>(
      self, size, strides, c10::nullopt, true);
  return self;
}

} // namespace native
} // namespace at
