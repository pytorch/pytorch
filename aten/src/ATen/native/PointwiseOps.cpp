// Ternary and higher-order pointwise operations
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/PointwiseOps.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(addcmul)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  build(TensorIteratorConfig()
      .allow_cpu_scalars(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .add_owned_output(maybe_get_output())
      .add_owned_const_input(self)
      .add_owned_const_input(tensor1)
      .add_owned_const_input(tensor2));
}

TORCH_META_FUNC(addcdiv)
(const Tensor& self,
 const Tensor& numerator,
 const Tensor& denominator,
 const Scalar& value) {
  if (isIntegralType(numerator.scalar_type(), /*includeBool=*/true) &&
      isIntegralType(denominator .scalar_type(), /*includeBool=*/true)) {
    TORCH_CHECK(
        false,
        "Integer division with addcdiv is no longer supported, and in a future  ",
        "release addcdiv will perform a true division of numerator and denominator. ",
        "The historic addcdiv behavior can be implemented as ",
        "(input + value * torch.trunc(numerator / denominator)).to(input.dtype) ",
        "for integer inputs and as ",
        "(input + value * numerator / denominator) for float inputs. ",
        "The future addcdiv behavior is just the latter implementation: ",
        "(input + value * numerator / denominator), for all dtypes.");
  }
  build(TensorIteratorConfig()
      .allow_cpu_scalars(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .add_owned_output(maybe_get_output())
      .add_owned_const_input(self)
      .add_owned_const_input(numerator)
      .add_owned_const_input(denominator));
}

} // namespace at::meta
namespace at::native {

TORCH_IMPL_FUNC(addcmul_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcmul_stub(device_type(), *this, value);
}

TORCH_IMPL_FUNC(addcdiv_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcdiv_stub(device_type(), *this, value);
}

DEFINE_DISPATCH(addcmul_stub);
DEFINE_DISPATCH(addcdiv_stub);

} // namespace at::native
