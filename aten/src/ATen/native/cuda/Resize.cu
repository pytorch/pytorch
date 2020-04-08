#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/cuda/Resize.cuh>
#include <ATen/native/ResizeCommon.h>

namespace at {
namespace native {
namespace {

Tensor& resize_cuda_(
    Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cuda_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}
static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<decltype(resize_cuda_), &resize_cuda_>(DispatchKey::CUDA))
  ;

} // namespace
} // namespace native
} // namespace at
