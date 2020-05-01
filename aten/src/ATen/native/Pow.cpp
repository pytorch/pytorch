#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>

namespace at { namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, base, exp,
                                        /*check_mem_overlap=*/true);
  pow_tensor_tensor_stub(iter.device_type(), iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& base, Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");
  // Avoid runtime error when typecasting
  if (!exp.isComplex() && (exp.toDouble() == 0.0)) {
    result.resize_as_(base).fill_(1);
  } else if (!exp.isComplex() && (exp.toDouble() == 1.0)) {
    result.resize_as_(base).copy_(base);
  } else {
    auto iter = TensorIterator::unary_op(result, base,
                                         /*check_mem_overlap=*/true);
    pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, Scalar base, const Tensor& exp) {
  if (base.toDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else {
    native::pow_out(result, c10::scalar_to_tensor(base, exp.device()), exp);
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
  // If the exponent is complex, the result needs to be complex
  // we can't rely on result_type because it will break current
  // handling
  // TODO: change it to use type promotion after #37098 is merged
  ScalarType dtype = (exp.is_complex() ? exp.scalar_type() : base.scalar_type());
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  return native::pow_out(result, base, exp);
}

Tensor pow(const Tensor& base, Scalar exp) {
  // If the exponent is complex, the result needs to be complex
  // we can't rely on result_type because it will break current
  // handling for other datatypes
  // TODO: change it to use type promotion after #37098 is merged
  ScalarType dtype = (exp.isComplex() ? exp.type() : base.scalar_type());
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  if (exp.isComplex()) {
    // The type checking logic in unary_op TensorIterator does not allow
    // a float tensor to output to a complex tensor, but binary ops allow it
    // so we create a tensor for the exponent to avoid using this iterator until its fixed
    return native::pow_out(result, base, c10::scalar_to_tensor(exp, base.device()));
  }
  return native::pow_out(result, base, exp);
}

Tensor pow(Scalar base, const Tensor& exp) {
  Tensor result = at::empty_like(exp, MemoryFormat::Preserve);
  return native::pow_out(result, base, exp);
}

} // namespace native

} // namespace at
