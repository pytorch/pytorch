#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);
DEFINE_DISPATCH(pow_scalar_tensor_stub);

Tensor& pow_out(Tensor& result, const Tensor& self, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, self, exp);
  pow_tensor_tensor_stub(iter.device_type(), iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& self, Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(self.scalar_type()) &&
              exp.isIntegral() && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");
  if (exp.toDouble() == 0.0) {
    result.copy_(ones(self.sizes(), self.options()));
  } else if (exp.toDouble() == 1.0) {
    result.copy_(self);
  } else {
    auto iter = TensorIterator::unary_op(result, self);
    pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, Scalar self, const Tensor& exp) {
  if (self.toDouble() == 1.0) {
    result.copy_(ones(exp.sizes(), exp.options()));
  } else {
    auto iter = TensorIterator();
    iter.add_output(result);
    iter.add_input(exp);
    iter.dont_compute_common_dtype();
    iter.build();
    pow_scalar_tensor_stub(iter.device_type(), iter, self);
  }
  return result;
}

Tensor& pow_(Tensor& self, const Tensor& other) {
  return native::pow_out(self, self, other);
}

Tensor& pow_(Tensor& self, Scalar alpha) {
  return native::pow_out(self, self, alpha);
}

Tensor pow(const Tensor& self, const Tensor& exp) {
  Tensor result = at::empty_like(self);
  return native::pow_out(result, self, exp);
}

Tensor pow(const Tensor& self, Scalar exp) {
  Tensor result = at::empty_like(self);
  return native::pow_out(result, self, exp);
}

Tensor pow(Scalar self, const Tensor& exp) {
  Tensor result = at::empty_like(exp,
    self.isFloatingPoint() ? ScalarType::Double : ScalarType::Long);
  return native::pow_out(result, self, exp);
}

} // namespace native

} // namespace at
