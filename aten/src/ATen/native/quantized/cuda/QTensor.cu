#include <ATen/ATen.h>
#include <ATen/native/Copy.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {

Tensor int_repr_quant_cuda(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIterator();
    iter.set_check_mem_overlap(false);
    iter.add_output(dst);
    iter.add_input(self);
    iter.dont_resize_outputs();
    iter.dont_compute_common_dtype();
    iter.build();
    gpu_kernel(iter, [](scalar_t value) -> underlying_t { return value.val_; });
  });
  return dst;
}

Tensor make_per_tensor_quantized_tensor_cuda(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst;
  auto qdtype = toQIntType(self.scalar_type());
  AT_CHECK(isQIntType(qdtype), "make_per_tensor_quantized_tensor works only with int8, uint8 and int32 dtypes right now.");
  AT_DISPATCH_QINT_TYPES(qdtype, "make_per_tensor_quantized_tensor", [&]() {
    dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(qdtype),
      scale,
      zero_point);
    auto iter = TensorIterator();
    iter.set_check_mem_overlap(false);
    iter.add_output(dst);
    iter.add_input(self);
    iter.dont_resize_outputs();
    iter.dont_compute_common_dtype();
    gpu_kernel(iter, [](underlying_t value) -> scalar_t { return scalar_t(value); });
  });
  return dst;
}

} // namespace native
} // namespace at
