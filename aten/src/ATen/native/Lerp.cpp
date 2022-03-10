#include <ATen/native/Lerp.h>
#include <ATen/NativeFunctions.h>

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

DEFINE_DISPATCH(lerp_kernel_scalar_weight);
DEFINE_DISPATCH(lerp_kernel_tensor_weight);

} // namespace native
} // namespace at
