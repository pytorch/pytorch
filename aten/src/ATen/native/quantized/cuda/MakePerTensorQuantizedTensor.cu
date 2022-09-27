#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cpu/vec/vec.h>

namespace at {
namespace native {

void assign_quantized_tensor_cuda(
  const Tensor& self, Tensor& dst) {
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "assign_quantized_tensor_cuda", [&]() {
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(dst)
          .add_input(self)
          .build();
        gpu_kernel(iter, [] GPU_LAMBDA(underlying_t value) -> scalar_t {
          return scalar_t(value);
        });
      });
}

Tensor make_per_tensor_quantized_tensor_cuda(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  assign_quantized_tensor_cuda(self, dst);
  return dst;
}

Tensor make_per_channel_quantized_tensor_cuda(
  const Tensor& self,
  const Tensor& scales,
  const Tensor& zero_points,
  int64_t axis) {
      Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  assign_quantized_tensor_cuda(self, dst);
  return dst;
}

Tensor& relu_quantized_cuda_(Tensor& self) {
  const auto zero_point = self.q_zero_point();
  AT_DISPATCH_QINT_TYPES(
    self.scalar_type(), "qrelu_cuda", [&]() {
      using Vec = vec::Vectorized<scalar_t>;
      auto iter = TensorIterator::unary_op(self, self);
      gpu_kernel(iter, [zero_point] GPU_LAMBDA(scalar_t value) -> scalar_t {
        return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        });
  });
  return self;
}

} // namespace native
} // namespace at
