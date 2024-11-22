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

  auto weight_ = weight;
  if (self.dtype() != weight.dtype()) {
    auto promote_type = c10::promoteTypes(self.scalar_type(), weight.scalar_type());
    TORCH_CHECK(promote_type == self.scalar_type(), "Unable to promote `input` dtype to ", promote_type,
                ", change `weight` dtype ", weight.dtype(), " same as `input` dtype ", self.dtype());
    weight_ = weight.to(promote_type);
  }

  build(at::TensorIteratorConfig()
        .allow_cpu_scalars(true)
        .add_output(maybe_get_output())
        .add_const_input(self)
        .add_const_input(end)
        .add_const_input(weight_));
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
