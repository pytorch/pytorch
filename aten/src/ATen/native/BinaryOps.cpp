#include <ATen/native/BinaryOps.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

#include <torch/library.h>

namespace at {
namespace meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

} // namespace meta


namespace native {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(add_clamp_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(div_floor_stub);
DEFINE_DISPATCH(div_trunc_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(logit_backward_stub);
DEFINE_DISPATCH(tanh_backward_stub);
DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);
DEFINE_DISPATCH(fmax_stub);
DEFINE_DISPATCH(fmin_stub);
DEFINE_DISPATCH(fmod_stub);
DEFINE_DISPATCH(logaddexp_stub);
DEFINE_DISPATCH(logaddexp2_stub);
DEFINE_DISPATCH(gcd_stub);
DEFINE_DISPATCH(lcm_stub);
DEFINE_DISPATCH(hypot_stub);
DEFINE_DISPATCH(igamma_stub);
DEFINE_DISPATCH(igammac_stub);
DEFINE_DISPATCH(nextafter_stub);
DEFINE_DISPATCH(heaviside_stub);
DEFINE_DISPATCH(copysign_stub);
DEFINE_DISPATCH(xlogy_stub);

static Tensor wrapped_scalar_tensor(const Scalar& scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

TORCH_IMPL_FUNC(add_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

Tensor& add_relu_impl(
    Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto iter = TensorIterator::binary_op(result, self, other);
  Scalar min_val;
  Scalar max_val;
  if (self.dtype() == at::kInt) {
    min_val = 0;
    max_val = std::numeric_limits<int32_t>::max();
  } else if (self.dtype() == at::kLong) {
    min_val = 0;
    max_val = std::numeric_limits<int64_t>::max();
  } else if (self.dtype() == at::kShort) {
    min_val = 0;
    max_val = std::numeric_limits<int16_t>::max();
  } else if (self.dtype() == at::kChar) {
    min_val = 0;
    max_val = std::numeric_limits<int8_t>::max();
  } else if (self.dtype() == at::kFloat) {
    min_val = 0.0;
    max_val = std::numeric_limits<float>::max();
  } else if (self.dtype() == at::kDouble) {
    min_val = 0.0;
    max_val = std::numeric_limits<double>::max();
  } else {
    TORCH_INTERNAL_ASSERT(
        "Unsupported datatype for add_relu:", self.dtype().name());
  }

  result = iter.output();
  add_clamp_stub(iter.device_type(), iter, alpha, min_val, max_val);
  return result;
}

Tensor& add_relu_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return add_relu_impl(result, self, other, alpha);
}

Tensor add_relu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor result;
  return add_relu_impl(result, self, other, alpha);
}

Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return add_relu_impl(self, self, other, alpha);
}

Tensor& copysign_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  copysign_stub(iter.device_type(), iter);
  return result;
}

Tensor copysign(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  copysign_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& copysign_(Tensor& self, const Tensor& other) {
  return native::copysign_out(self, other, self);
}

Tensor copysign(const Tensor& self, const Scalar& other) {
  return native::copysign(self, wrapped_scalar_tensor(other));
}

Tensor& copysign_(Tensor& self, const Scalar& other) {
  return native::copysign_(self, wrapped_scalar_tensor(other));
}

Tensor& div_true_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  div_true_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor& div_trunc_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_trunc_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor& div_floor_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor& div_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return div_true_out(self, other, result);
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  div_true_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return div_true_out(self, other, self);
}

// WARNING: There doesn't appear to be any testing for this function
// with sparse self input.
Tensor div(const Tensor& self, const Scalar& other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

// WARNING: This function, with a sparse self, is currently only
// exercised by DistributedDataParallelTest.test_sparse_gradients
// (you need to exercise it from C++, because this overload is never
// used for Python)
Tensor& div_(Tensor& self, const Scalar& other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor& div_out(const Tensor& self, const Tensor& other, std::string rounding_mode, Tensor& result) {
  if (rounding_mode == "true") {
    return div_true_out(self, other, result);
  } else if (rounding_mode == "trunc") {
    return div_trunc_out(self, other, result);
  } else if (rounding_mode == "floor") {
    return div_floor_out(self, other, result);
  }

  TORCH_CHECK(false,
      "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
      "but found '", rounding_mode, "'");
}

Tensor div(const Tensor& self, const Tensor& other, std::string rounding_mode) {
  Tensor result;
  native::div_out(self, other, std::move(rounding_mode), result);
  return result;
}

Tensor& div_(Tensor& self, const Tensor& other, std::string rounding_mode) {
  return native::div_out(self, other, std::move(rounding_mode), self);
}

Tensor div(const Tensor& self, const Scalar& other, std::string rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

Tensor& div_(Tensor& self, const Scalar& other, std::string rounding_mode) {
  return self.div_(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

// divide, alias for div
Tensor& divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::div_out(result, self, other);
}

Tensor divide(const Tensor& self, const Tensor& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Tensor& other) {
  return self.div_(other);
}

Tensor divide(const Tensor& self, const Scalar& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Scalar& other) {
  return self.div_(other);
}

Tensor& divide_out(const Tensor& self, const Tensor& other, std::string rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Tensor& other, std::string rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Tensor& other, std::string rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Scalar& other, std::string rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Scalar& other, std::string rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

// true_divide, an alias for div
Tensor& true_divide_out(const Tensor& self, const Tensor& divisor, Tensor& result) {
  return at::div_out(result, self, divisor);
}

Tensor true_divide(const Tensor& self, const Tensor& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Tensor& divisor) {
  return self.div_(divisor);
}

Tensor true_divide(const Tensor& self, const Scalar& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Scalar& divisor) {
  return self.div_(divisor);
}

Tensor& remainder_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  remainder_stub(iter.device_type(), iter);
  return result;
}

Tensor remainder(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  remainder_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& remainder_(Tensor& self, const Tensor& other) {
  return native::remainder_out(self, self, other);
}

Tensor& floor_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_WARN_ONCE(
    "floor_divide is deprecated, and will be removed in a future version of pytorch. "
    "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
    "This results in incorrect rounding for negative values.\n"
    "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
    "or for actual floor division, use torch.div(a, b, rounding_mode='floor')."
  );
  // FIXME: Not actually doing floor division (#43874)
  return div_trunc_out(self, other, result);
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  TORCH_WARN_ONCE(
    "floor_divide is deprecated, and will be removed in a future version of pytorch. "
    "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
    "This results in incorrect rounding for negative values.\n"
    "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
    "or for actual floor division, use torch.div(a, b, rounding_mode='floor')."
  );
  // FIXME: Not actually doing floor division (#43874)
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_trunc_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return native::floor_divide_out(self, other, self);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter.device_type(), iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

Tensor mul(const Tensor& self, const Scalar& other) {
  return at::mul(self, wrapped_scalar_tensor(other)); // redispatch!
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::mul_out(self, self, wrapped_scalar_tensor(other)); // redispatch!
}

// multiply, alias for mul
Tensor& multiply_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::mul_out(result, self, other);
}

Tensor multiply(const Tensor& self, const Tensor& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Tensor& other) {
  return self.mul_(other);
}

Tensor multiply(const Tensor& self, const Scalar& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Scalar& other) {
  return self.mul_(other);
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::sub_out(self, self, other, alpha);
}

Tensor sub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::sub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::sub_(self, wrapped_scalar_tensor(other), alpha);
}

// subtract, alias for sub
Tensor& subtract_out(Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::sub_out(result, self, other, alpha);
}

Tensor subtract(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor subtract(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor& sigmoid_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& logit_backward_out(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& input,
    c10::optional<double> eps) {
  auto iter = TensorIterator::binary_op(result, grad_output, input);
  logit_backward_stub(
      iter.device_type(), iter, Scalar(eps ? eps.value() : -1.0));
  return result;
}

Tensor logit_backward(
    const Tensor& grad_output,
    const Tensor& input,
    c10::optional<double> eps) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, input);
  logit_backward_stub(
      iter.device_type(), iter, Scalar(eps ? eps.value() : -1.0));
  return iter.output();
}

Tensor& tanh_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::sub(other, self, alpha);
}

Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_stub(iter.device_type(), iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return native::atan2_out(self, self, other);
}

// These are still needed because we don't have C++ conversions from number
// types (int, float, etc.) to Tensor (only to Scalar). They're not exposed
// to Python.

static void check_convert(const Scalar& scalar, ScalarType scalarType) {
  // Validate that is possible to convert scalar to tensor dtype without overflow
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, scalarType, "check_convert", [&]{
    scalar.to<scalar_t>();
  });
}

static Tensor wrapped_scalar_tensor_and_check_convert(const Scalar& scalar, Tensor tensor) {
  check_convert(scalar, tensor.scalar_type());
  return wrapped_scalar_tensor(scalar);
}

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here

Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}

Tensor remainder(const Tensor& self, const Scalar& other) {
  return native::remainder(self, wrapped_scalar_tensor(other));
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  return native::remainder_(self, wrapped_scalar_tensor(other));
}

Tensor& remainder_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return native::remainder_out(result, self, wrapped_scalar_tensor(other));
}

Tensor rsub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  bitwise_and_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_and(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_and_out(result, self, other);
  return result;
}

Tensor& bitwise_and_(Tensor& self, const Tensor& other) {
  return at::bitwise_and_out(self, self, other);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return at::bitwise_and_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_and(const Tensor& self, const Scalar& other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_and_out(result, self, other);
}

Tensor& bitwise_and_(Tensor& self, const Scalar& other) {
  return at::bitwise_and_out(self, self, other);
}

// Legacy and interfaces. They are aliased to bitwise_and* functions
Tensor __and__(const Tensor& self, const Tensor& other) {
  return at::bitwise_and(self, other);
}

Tensor __and__(const Tensor& self, const Scalar& other) {
  return at::bitwise_and(self, other);
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  return self.bitwise_and_(other);
}

Tensor& __iand__(Tensor& self, const Scalar& other) {
  return self.bitwise_and_(other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  bitwise_or_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_or(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_or_out(result, self, other);
  return result;
}

Tensor& bitwise_or_(Tensor& self, const Tensor& other) {
  return at::bitwise_or_out(self, self, other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return at::bitwise_or_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_or(const Tensor& self, const Scalar& other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_or_out(result, self, other);
}

Tensor& bitwise_or_(Tensor& self, const Scalar& other) {
  return at::bitwise_or_out(self, self, other);
}

// Legacy or interfaces. They are aliased to bitwise_or* functions
Tensor __or__(const Tensor& self, const Tensor& other) {
  return at::bitwise_or(self, other);
}

Tensor __or__(const Tensor& self, const Scalar& other) {
  return at::bitwise_or(self, other);
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  return self.bitwise_or_(other);
}

Tensor& __ior__(Tensor& self, const Scalar& other) {
  return self.bitwise_or_(other);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  bitwise_xor_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_xor(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_xor_out(result, self, other);
  return result;
}

Tensor& bitwise_xor_(Tensor& self, const Tensor& other) {
  return at::bitwise_xor_out(self, self, other);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, const Scalar& other) {
  return at::bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_xor(const Tensor& self, const Scalar& other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_xor_out(result, self, other);
}

Tensor& bitwise_xor_(Tensor& self, const Scalar& other) {
  return at::bitwise_xor_out(self, self, other);
}

// Legacy xor interfaces. They are aliased to bitwise_xor* functions
Tensor __xor__(const Tensor& self, const Tensor& other) {
  return at::bitwise_xor(self, other);
}

Tensor __xor__(const Tensor& self, const Scalar& other) {
  return at::bitwise_xor(self, other);
}

Tensor& __ixor__(Tensor& self, const Tensor& other) {
  return self.bitwise_xor_(other);
}

Tensor& __ixor__(Tensor& self, const Scalar& other) {
  return self.bitwise_xor_(other);
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __lshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __ilshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __rshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  rshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __irshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return self;
}

template <typename Stub>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  // Validate that is possible to convert zero-dim tensor's dtype to other dtype without overflow
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

// To avoid overflow during type promotion we will check that both dtypes of self and other are same
template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  TORCH_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  return out_impl(self, self, other);
}

// validates that is possible to convert Scalar other to self's dtype without overflow.
// This behavior is unique to comparison ops; arithmetic operations don't do this.
// In the future, we should reconsider this inconsistency and decide if we want to add the same check to arithmetic ops.
template <typename OutImpl>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor_and_check_convert(other, self), out_impl);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguious.
using OutFunc = std::add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;

Tensor& lt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, lt_stub); }
Tensor lt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor lt(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }

// less, alias for torch.lt
Tensor& less_out(Tensor& result, const Tensor& self, const Tensor& other) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Tensor& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Tensor& other) { return self.lt_(other); }
Tensor& less_out(Tensor& result, const Tensor& self, const Scalar& other) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Scalar& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Scalar& other) { return self.lt_(other); }

Tensor& le_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, le_stub); }
Tensor le(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::le_out)); }
Tensor le(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }

// less_equal, alias for torch.le
Tensor& less_equal_out(Tensor& result, const Tensor& self, const Tensor& other) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Tensor& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Tensor& other) { return self.le_(other); }
Tensor& less_equal_out(Tensor& result, const Tensor& self, const Scalar& other) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Scalar& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Scalar& other) { return self.le_(other); }

Tensor& gt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, gt_stub); }
Tensor gt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor gt(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }

// greater, alias for torch.gt
Tensor& greater_out(Tensor& result, const Tensor& self, const Tensor& other) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Tensor& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Tensor& other) { return self.gt_(other); }
Tensor& greater_out(Tensor& result, const Tensor& self, const Scalar& other) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Scalar& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Scalar& other) { return self.gt_(other); }

Tensor& ge_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ge_stub); }
Tensor ge(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor ge(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }

// greater_equal, alias for torch.ge
Tensor& greater_equal_out(Tensor& result, const Tensor& self, const Tensor& other) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Tensor& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Tensor& other) { return self.ge_(other); }
Tensor& greater_equal_out(Tensor& result, const Tensor& self, const Scalar& other) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Scalar& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Scalar& other) { return self.ge_(other); }

Tensor& eq_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, eq_stub); }
Tensor eq(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor eq(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }

Tensor& ne_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ne_stub); }
Tensor ne(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor ne(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }

// not_equal, alias for torch.ne
Tensor& not_equal_out(Tensor& result, const Tensor& self, const Tensor& other) { return at::ne_out(result, self, other); }
Tensor not_equal(const Tensor& self, const Tensor& other) { return self.ne(other); }
Tensor& not_equal_(Tensor& self, const Tensor& other) { return self.ne_(other); }
Tensor& not_equal_out(Tensor& result, const Tensor& self, const Scalar& other) { return at::ne_out(result, self, other); }
Tensor not_equal(const Tensor& self, const Scalar& other) { return self.ne(other); }
Tensor& not_equal_(Tensor& self, const Scalar& other) { return self.ne_(other); }

Tensor& logical_and_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_and_stub); }
Tensor logical_and(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor logical_and(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }

Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_or_stub); }
Tensor logical_or(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor logical_or(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }

Tensor& logical_xor_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_xor_stub); }
Tensor logical_xor(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor logical_xor(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }

Tensor& maximum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  maximum_stub(iter.device_type(), iter);
  return result;
}

Tensor maximum(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  maximum_stub(iter.device_type(), iter);
  return iter.output();
}

// binary max, alias for maximum
Tensor& max_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::maximum_out(result, self, other);
}

Tensor max(const Tensor& self, const Tensor& other) {
  return at::maximum(self, other);
}

Tensor& fmax_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");

  auto iter = TensorIterator::binary_op(result, self, other);
  fmax_stub(iter.device_type(), iter);
  return result;
}

Tensor fmax(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  fmax_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& minimum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  minimum_stub(iter.device_type(), iter);
  return result;
}

Tensor minimum(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  minimum_stub(iter.device_type(), iter);
  return iter.output();
}

// binary min, alias for minimum
Tensor& min_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::minimum_out(result, self, other);
}

Tensor min(const Tensor& self, const Tensor& other) {
  return at::minimum(self, other);
}

Tensor& fmin_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");

  auto iter = TensorIterator::binary_op(result, self, other);
  fmin_stub(iter.device_type(), iter);
  return result;
}

Tensor fmin(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  fmin_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor floor_divide(const Tensor& self, const Scalar& other) {
  return at::floor_divide(self, wrapped_scalar_tensor(other));
}

Tensor& floor_divide_(Tensor& self, const Scalar& other) {
  return at::floor_divide_out(self, self, wrapped_scalar_tensor(other));
}

Tensor& fmod_out(Tensor & result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  fmod_stub(iter.device_type(), iter);
  return result;
}

Tensor& fmod_out(Tensor & result, const Tensor& self, const Scalar& other) {
  return native::fmod_out(result, self, wrapped_scalar_tensor(other));
}

Tensor fmod(const Tensor& self, const Tensor & other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  fmod_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor fmod(const Tensor& self, const Scalar& other) {
  return native::fmod(self, wrapped_scalar_tensor(other));
}

Tensor& fmod_(Tensor& self, const Tensor& other) {
  return native::fmod_out(self, self, other);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return native::fmod_(self, wrapped_scalar_tensor(other));
}

Tensor& logaddexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  logaddexp_stub(iter.device_type(), iter);
  return result;
}

Tensor logaddexp(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::logaddexp_out(result, self, other);
}

Tensor& logaddexp2_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  logaddexp2_stub(iter.device_type(), iter);
  return result;
}

Tensor logaddexp2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::logaddexp2_out(result, self, other);
}

Tensor& gcd_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  gcd_stub(iter.device_type(), iter);
  return result;
}

Tensor gcd(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::gcd_out(result, self, other);
}

Tensor& gcd_(Tensor& self, const Tensor& other) {
  return at::gcd_out(self, self, other);
}

Tensor& lcm_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  lcm_stub(iter.device_type(), iter);
  return result;
}

Tensor lcm(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::lcm_out(result, self, other);
}

Tensor& lcm_(Tensor& self, const Tensor& other) {
  return at::lcm_out(self, self, other);
}

Tensor& hypot_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  hypot_stub(iter.device_type(), iter);
  return result;
}

Tensor hypot(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  hypot_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& hypot_(Tensor& self, const Tensor& other) {
  return at::hypot_out(self, self, other);
}

Tensor& igamma_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  igamma_stub(iter.device_type(), iter);
  return result;
}

Tensor igamma(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  igamma_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& igamma_(Tensor& self, const Tensor& other) {
  return at::igamma_out(self, self, other);
}

Tensor& igammac_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  igammac_stub(iter.device_type(), iter);
  return result;
}

Tensor igammac(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  igammac_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& igammac_(Tensor& self, const Tensor& other) {
  return at::igammac_out(self, self, other);
}

Tensor& nextafter_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  nextafter_stub(iter.device_type(), iter);
  return result;
}

Tensor nextafter(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  nextafter_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& nextafter_(Tensor& self, const Tensor& other) {
  return at::nextafter_out(self, self, other);
}

// Note: this function is only for testing.
// It is undocumented and should not be used outside of tests.
Tensor _test_serialization_subcmul(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self - (other * alpha);
}

Tensor& heaviside_out(Tensor& result, const Tensor& self, const Tensor& values) {
  TORCH_CHECK(!self.is_complex() && !result.is_complex() && !values.is_complex(),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.dtype() == values.dtype() &&  result.dtype() == self.dtype(),
              "heaviside is not yet implemented for tensors with different dtypes.");

  auto iter = TensorIterator::binary_op(result, self, values);
  heaviside_stub(iter.device_type(), iter);
  return result;
}

Tensor heaviside(const Tensor& self, const Tensor& values) {
  TORCH_CHECK(!self.is_complex() && !values.is_complex(),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.dtype() == values.dtype(),
              "heaviside is not yet implemented for tensors with different dtypes.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, values);
  heaviside_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& heaviside_(Tensor& self, const Tensor& values) {
  return at::heaviside_out(self, self, values);
}

Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, at::pow(2.0, other));
}

Tensor ldexp(const Tensor& self, const Tensor& other) {
  return at::mul(self, at::pow(2.0, other));
}

Tensor& ldexp_(Tensor& self, const Tensor& other) {
  return at::ldexp_out(self, self, other);
}

Tensor& xlogy_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  xlogy_stub(iter.device_type(), iter);
  return result;
}

Tensor& xlogy_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::xlogy_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& xlogy_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::xlogy_out(result, self, wrapped_scalar_tensor(other));
}

Tensor xlogy(const Tensor& x, const Tensor& y) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, x, y);
  xlogy_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor xlogy(const Scalar& x, const Tensor& y) {
  return at::xlogy(wrapped_scalar_tensor(x), y);
}

Tensor xlogy(const Tensor& x, const Scalar& y) {
  return at::xlogy(x, wrapped_scalar_tensor(y));
}

Tensor& xlogy_(Tensor& x, const Tensor& y) {
  return at::xlogy_out(x, x, y);
}

Tensor& xlogy_(Tensor& x, const Scalar& y) {
  return at::xlogy_out(x, x, wrapped_scalar_tensor(y));
}

} // namespace native
} // namespace at
