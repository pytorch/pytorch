
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Resize.cuh>
#include <ATen/core/op_registration/op_registration.h>

namespace at {
namespace native {

namespace {
// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_empty_(Tensor& result) {
  Storage storage(result.dtype(), 0, at::cuda::getCUDADeviceAllocator(), true);
  return result.set_(storage, 0, {0}, {});
}

static auto registry_set_cuda_ = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("set_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<decltype(set_empty_), &set_empty_>(DispatchKey::CUDATensorId))
  ;

// unify with cuda implementation?  This is not done to avoid a dispatch in resize_impl_cpu_
Tensor& set_storage_cuda_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) { // Note-to-self: This handles all CUDA-only implementations
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          c10::optional<IntArrayRef>(stride) : c10::nullopt;
  at::native::resize_impl_cuda_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

static auto registry_set_storage_ = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<decltype(set_storage_cuda_), &set_storage_cuda_>(DispatchKey::CUDATensorId))
  ;

}
}
}
