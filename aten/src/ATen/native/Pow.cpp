#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);
DEFINE_DISPATCH(pow_scalar_tensor_stub);

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, base, exp);
  pow_tensor_tensor_stub(iter.device_type(), iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& base, Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type()) &&
              exp.isIntegral() && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");
  if (exp.toDouble() == 0.0) {
    result.copy_(ones(base.sizes(), base.options()));
  } else if (exp.toDouble() == 1.0) {
    result.copy_(base);
  } else {
    auto iter = TensorIterator::unary_op(result, base);
    pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, Scalar base, const Tensor& exp) {
  if (base.toDouble() == 1.0) {
    result.copy_(ones(exp.sizes(), exp.options()));
  } else {
    auto iter = TensorIterator::unary_op(result, exp);
    pow_scalar_tensor_stub(iter.device_type(), iter, base);
  }
  return result;
}

Tensor& pow_(Tensor& base, const Tensor& other) {
  return native::pow_out(base, base, other);
}

Tensor& pow_(Tensor& base, Scalar alpha) {
  return native::pow_out(base, base, alpha);
}

Tensor pow(const Tensor& base, const Tensor& exp) {
  Tensor result = at::empty_like(base);
  return native::pow_out(result, base, exp);
}

Tensor pow(const Tensor& base, Scalar exp) {
  Tensor result = at::empty_like(base);
  return native::pow_out(result, base, exp);
}

Tensor pow(Scalar base, const Tensor& exp) {
  Tensor result = at::empty_like(exp);
  return native::pow_out(result, base, exp);
}

} // namespace native

} // namespace at
