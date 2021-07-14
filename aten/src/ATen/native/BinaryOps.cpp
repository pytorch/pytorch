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
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(sub, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  native::sub_check(self, other);
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(mul, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (*rounding_mode == "trunc") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "floor") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else {
    TORCH_CHECK(false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '", *rounding_mode, "'");
  }
}

TORCH_META_FUNC(special_xlog1py) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(special_zeta) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(copysign, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(heaviside) (
  const Tensor& self, const Tensor& other
) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex() &&
              (maybe_get_output().defined() ? !maybe_get_output().is_complex() : true),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.dtype() == other.dtype() &&
              (maybe_get_output().defined() ? maybe_get_output().dtype() == self.dtype() : true),
              "heaviside is not yet implemented for tensors with different dtypes.");

  build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(atan2) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(remainder, Tensor)(const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_left_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_right_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

// These are normal binary ops that preserve dtype
#define CREATE_BINARY_META_FUNC(func)                                 \
  TORCH_META_FUNC(func) (const Tensor& self, const Tensor& other) {   \
    build_borrowing_binary_op(maybe_get_output(), self, other);                 \
  }

CREATE_BINARY_META_FUNC(logaddexp);
CREATE_BINARY_META_FUNC(logaddexp2);
CREATE_BINARY_META_FUNC(gcd);
CREATE_BINARY_META_FUNC(lcm);
CREATE_BINARY_META_FUNC(hypot);
CREATE_BINARY_META_FUNC(igamma);
CREATE_BINARY_META_FUNC(igammac);
CREATE_BINARY_META_FUNC(nextafter);

TORCH_META_FUNC(maximum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(minimum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmax) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
    build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmin) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
    build_binary_op(maybe_get_output(), self, other);
}

} // namespace meta


namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(add_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(add_clamp_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sub_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(mul_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(div_true_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(div_floor_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(div_trunc_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(remainder_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atan2_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_and_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_or_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_xor_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(lshift_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(rshift_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_and_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_or_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_xor_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(lt_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(le_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(gt_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ge_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(eq_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ne_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sigmoid_backward_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logit_backward_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(tanh_backward_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(maximum_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(minimum_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(fmax_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(fmin_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(fmod_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logaddexp_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logaddexp2_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(gcd_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(lcm_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(hypot_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(igamma_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(igammac_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(nextafter_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(heaviside_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(copysign_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(xlogy_stub);
DEFINE_DISPATCH(xlog1py_stub);
DEFINE_DISPATCH(zeta_stub);

TORCH_IMPL_FUNC(add_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

TORCH_IMPL_FUNC(sub_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  sub_stub(device_type(), *this, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  mul_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  div_true_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out_mode) (
  const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    div_floor_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(special_xlog1py_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  xlog1py_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_zeta_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  zeta_stub(device_type(), *this);
}

#define CREATE_BINARY_TORCH_IMPL_FUNC(func_out, func_stub)                                                    \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& other, const Tensor& result) {  \
  func_stub(device_type(), *this);                                                           \
}

CREATE_BINARY_TORCH_IMPL_FUNC(maximum_out, maximum_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(minimum_out, minimum_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmax_out, fmax_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmin_out, fmin_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp2_out, logaddexp2_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(gcd_out, gcd_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(lcm_out, lcm_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igamma_out, igamma_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igammac_out, igammac_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(remainder_out, remainder_stub);

Tensor special_xlog1py(const Scalar& x, const Tensor& y) {
  return at::special_xlog1py(wrapped_scalar_tensor(x), y);
}

Tensor special_xlog1py(const Tensor& x, const Scalar& y) {
  return at::special_xlog1py(x, wrapped_scalar_tensor(y));
}

Tensor& special_xlog1py_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_xlog1py_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& special_xlog1py_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_xlog1py_out(result, self, wrapped_scalar_tensor(other));
}

Tensor special_zeta(const Scalar& x, const Tensor& y) {
  return at::special_zeta(wrapped_scalar_tensor(x), y);
}

Tensor special_zeta(const Tensor& x, const Scalar& y) {
  return at::special_zeta(x, wrapped_scalar_tensor(y));
}

Tensor& special_zeta_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_zeta_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& special_zeta_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_zeta_out(result, self, wrapped_scalar_tensor(other));
}

TORCH_IMPL_FUNC(atan2_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  atan2_stub(device_type(), *this);
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

Tensor add_relu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return add_relu_impl(self, self, other, alpha);
}

Tensor& add_relu_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu_(self, wrapped_scalar_tensor(other), alpha);
}

TORCH_IMPL_FUNC(copysign_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  copysign_stub(device_type(), *this);
}

Tensor copysign(const Tensor& self, const Scalar& other) {
  // redispatch!
  return at::copysign(self, wrapped_scalar_tensor(other));
}

Tensor& copysign_(Tensor& self, const Scalar& other) {
  // redispatch!
  return self.copysign_(wrapped_scalar_tensor(other));
}

Tensor& copysign_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch!
  return at::copysign_out(result, self, wrapped_scalar_tensor(other));
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

Tensor div(const Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

Tensor& div_(Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
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

Tensor& divide_out(const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
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

Tensor& floor_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_WARN_ONCE(
    "floor_divide is deprecated, and will be removed in a future version of pytorch. "
    "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
    "This results in incorrect rounding for negative values.\n"
    "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
    "or for actual floor division, use torch.div(a, b, rounding_mode='floor')."
  );
  // FIXME: Not actually doing floor division (#43874)
  auto iter = TensorIterator::binary_op(result, self, other);
  div_trunc_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
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

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here
Tensor mul(const Tensor& self, const Scalar& other) {
  return at::mul(self, wrapped_scalar_tensor(other)); // redispatch!
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::mul_out(self, wrapped_scalar_tensor(other), self); // redispatch!
}

// multiply, alias for mul
Tensor& multiply_out(const Tensor& self, const Tensor& other, Tensor& result) {
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

Tensor sub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::sub(self, wrapped_scalar_tensor(other), alpha); // redispatch!
}

Tensor& sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(wrapped_scalar_tensor(other), alpha); // redispatch!
}

// subtract, alias for sub
Tensor& subtract_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
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

Tensor& sigmoid_backward_out(const Tensor& grad_output, const Tensor& output, Tensor& result) {
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

Tensor& logit_backward_out(const Tensor& grad_output,
    const Tensor& input,
    c10::optional<double> eps,
    Tensor& result) {
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

Tensor& tanh_backward_out(const Tensor& grad_output, const Tensor& output, Tensor& result) {
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
  return at::sub(other, self, alpha); // redispatch!
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
  // redispatch
  return at::remainder(self, wrapped_scalar_tensor(other));
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.remainder_(wrapped_scalar_tensor(other));
}

Tensor& remainder_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch
  return at::remainder_out(result, self, wrapped_scalar_tensor(other));
}

Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(wrapped_scalar_tensor(self), other);
}

Tensor rsub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(const Tensor& self, const Tensor& other, Tensor& result) {
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

Tensor& bitwise_and_out(const Tensor& self, const Scalar& other, Tensor& result) {
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

Tensor& bitwise_or_out(const Tensor& self, const Tensor& other, Tensor& result) {
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

Tensor& bitwise_or_out(const Tensor& self, const Scalar& other, Tensor& result) {
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

Tensor& bitwise_xor_out(const Tensor& self, const Tensor& other, Tensor& result) {
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

Tensor& bitwise_xor_out(const Tensor& self, const Scalar& other, Tensor& result) {
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

TORCH_IMPL_FUNC(bitwise_left_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  lshift_stub(device_type(), *this);
}

Tensor& bitwise_left_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_left_shift_out(result, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_left_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift(self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor& bitwise_left_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift_out(self, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_left_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_left_shift(wrapped_scalar_tensor(self).toType(other.scalar_type()), other);
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

TORCH_IMPL_FUNC(bitwise_right_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  rshift_stub(device_type(), *this);
}

Tensor& bitwise_right_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_right_shift_out(result, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_right_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift(self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor& bitwise_right_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift_out(self, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_right_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_right_shift(wrapped_scalar_tensor(self).toType(other.scalar_type()), other);
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

Tensor& lt_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, lt_stub); }
Tensor lt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor lt(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }

// less, alias for torch.lt
Tensor& less_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Tensor& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Tensor& other) { return self.lt_(other); }
Tensor& less_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Scalar& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Scalar& other) { return self.lt_(other); }

Tensor& le_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, le_stub); }
Tensor le(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::le_out)); }
Tensor le(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }

// less_equal, alias for torch.le
Tensor& less_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Tensor& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Tensor& other) { return self.le_(other); }
Tensor& less_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Scalar& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Scalar& other) { return self.le_(other); }

Tensor& gt_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, gt_stub); }
Tensor gt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor gt(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }

// greater, alias for torch.gt
Tensor& greater_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Tensor& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Tensor& other) { return self.gt_(other); }
Tensor& greater_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Scalar& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Scalar& other) { return self.gt_(other); }

Tensor& ge_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, ge_stub); }
Tensor ge(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor ge(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }

// greater_equal, alias for torch.ge
Tensor& greater_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Tensor& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Tensor& other) { return self.ge_(other); }
Tensor& greater_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Scalar& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Scalar& other) { return self.ge_(other); }

Tensor& eq_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, eq_stub); }
Tensor eq(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor eq(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }

Tensor& ne_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, ne_stub); }
Tensor ne(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_out(const Tensor& self, const Scalar& other, Tensor& result) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor ne(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }

// not_equal, alias for torch.ne
Tensor& not_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ne_out(result, self, other); }
Tensor not_equal(const Tensor& self, const Tensor& other) { return self.ne(other); }
Tensor& not_equal_(Tensor& self, const Tensor& other) { return self.ne_(other); }
Tensor& not_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::ne_out(result, self, other); }
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

// binary max, alias for maximum
Tensor& max_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::maximum_out(result, self, other);
}

Tensor max(const Tensor& self, const Tensor& other) {
  return at::maximum(self, other);
}

// binary min, alias for minimum
Tensor& min_out(const Tensor& self, const Tensor& other, Tensor& result) {
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

Tensor& fmod_out(const Tensor& self, const Tensor& other, Tensor & result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  fmod_stub(iter.device_type(), iter);
  return result;
}

Tensor& fmod_out(const Tensor& self, const Scalar& other, Tensor & result) {
  return native::fmod_out(self, wrapped_scalar_tensor(other), result);
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
  return native::fmod_out(self, other, self);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return native::fmod_(self, wrapped_scalar_tensor(other));
}

// Note: this function is only for testing.
// It is undocumented and should not be used outside of tests.
Tensor _test_serialization_subcmul(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self - (other * alpha);
}

TORCH_IMPL_FUNC(heaviside_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  heaviside_stub(device_type(), *this);
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
