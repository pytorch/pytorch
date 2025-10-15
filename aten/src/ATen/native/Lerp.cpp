#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/lerp_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(lerp_Tensor)(
    const Tensor& self, const Tensor& end, const Tensor& weight) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  bool promote_weight = weight.dim() == 0;
  if (!promote_weight) {
    TORCH_CHECK(self.dtype() == weight.dtype(), "expected dtype ", self.dtype(),
                " for `weight` but got dtype ", weight.dtype());
  }
  build(at::TensorIteratorConfig()
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(promote_weight)
        .enforce_safe_casting_to_output(promote_weight)
        .cast_common_dtype_to_outputs(promote_weight)
        .add_output(maybe_get_output())
        .add_const_input(self)
        .add_const_input(end)
        .add_const_input(weight));
}

TORCH_META_FUNC(lerp_Scalar)(
    const Tensor& self, const Tensor& end, const Scalar& /*weight*/) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  build_binary_op(maybe_get_output(), self, end);
}

}  // namespace at::meta

namespace at::native {

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

} // namespace at::native
