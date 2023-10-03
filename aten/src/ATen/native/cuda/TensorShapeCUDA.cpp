#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/set_native.h>
#endif

#include <iostream>

namespace at::native {

// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cuda_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::cuda::getCUDADeviceAllocator(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& resize_storage_cuda_(Tensor& result, int64_t s) {
  c10::cuda::device_synchronize();
  std::cout << "resize_storage_cuda_" << std::endl;
  auto size_bytes = s * elementSize(result.scalar_type());
  std::cout << "size_bytes" << std::endl;
  const auto new_size_bytes = static_cast<size_t>(size_bytes);
  std::cout << "new_size_bytes" << std::endl;
  const Storage &storage = result.unsafeGetTensorImpl()->unsafe_storage();
  std::cout << "storage" << std::endl;
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  std::cout << "resize time" << std::endl;
  resize_bytes_cuda(storage.unsafeGetStorageImpl(), new_size_bytes);
  std::cout << "done resize" << std::endl;
  c10::cuda::device_synchronize();
  return result;
}

// unify with cuda implementation?  This is not done to avoid a dispatch in resize_impl_cpu_
Tensor& set_storage_cuda_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride, bool cross_device) {
  checkSetStorage(result, storage, storage_offset, size, stride, cross_device);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  at::native::resize_impl_cuda_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

} // namespace at::native
