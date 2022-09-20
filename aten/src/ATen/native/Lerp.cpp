#include <ATen/core/Tensor.h>
#include <ATen/native/Lerp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#else
#include <ATen/ops/lerp_native.h>
#include <ATen/ops/lerp.h>
#include <ATen/ops/_lerp_scalar_native.h>
#endif

namespace at {
namespace meta {

TORCH_META_FUNC(lerp_Tensor)(
    const Tensor& self, const Tensor& end, const Tensor& weight) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  TORCH_CHECK(self.dtype() == weight.dtype(), "expected dtype ", self.dtype(),
              " for `weight` but got dtype ", weight.dtype());
  build(at::TensorIteratorConfig()
        .add_output(maybe_get_output())
        .add_input(self)
        .add_input(end)
        .add_input(weight));
}

TORCH_META_FUNC(lerp_Scalar)(
    const Tensor& self, const Tensor& end, const Scalar& /*weight*/) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  build_binary_op(maybe_get_output(), self, end);
}

TORCH_META_FUNC(_lerp_scalar)(
    const Scalar& /*start*/, const Scalar& /*end*/, const Tensor& weight) {
  build_unary_op(maybe_get_output(), weight);
}

}  // namespace meta

namespace native {

TORCH_IMPL_FUNC(lerp_Tensor)(
    const Tensor& /*self*/, const Tensor& /*end*/, const Tensor& weight, const Tensor& /*out*/) {
  lerp_kernel_tensor_weight(device_type(), *this);
}

TORCH_IMPL_FUNC(lerp_Scalar)(
    const Tensor& /*self*/, const Tensor& /*end*/, const Scalar& weight, const Tensor& /*out*/) {
  lerp_kernel_scalar_weight(device_type(), *this, weight);
}

TORCH_IMPL_FUNC(_lerp_scalar)(
    const Scalar& start, const Scalar& end, const Tensor& /*weight*/, const Tensor& /*out*/) {
  lerp_kernel_scalar_start_end_stub(device_type(), *this, start, end);
}

DEFINE_DISPATCH(lerp_kernel_scalar_weight);
DEFINE_DISPATCH(lerp_kernel_tensor_weight);
DEFINE_DISPATCH(lerp_kernel_scalar_start_end_stub);

} // namespace native
} // namespace at
