#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BetaOps.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMeta.h>
#include <c10/util/MathConstants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tensor.h>
#include <ATen/ops/special_betainc.h>
#include <ATen/ops/special_betainc_native.h>
#include <ATen/ops/special_betaln.h>
#include <ATen/ops/special_betaln_native.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/minimum_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/reciprocal_native.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log_native.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/lgamma_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/where_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/exp_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/special_ndtri.h>
#include <ATen/ops/special_ndtri_native.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_like_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/xlogy_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/special_xlog1py.h>
#include <ATen/ops/special_xlog1py_native.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/equal_native.h>
#include <ATen/ops/square.h>
#include <ATen/ops/square_native.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/eq_native.h>

#endif

namespace at::meta {

TORCH_META_FUNC(special_betainc) (const Tensor& self, const Tensor& a, const Tensor& b) {
  build(TensorIteratorConfig()
    .allow_cpu_scalars(true) // same as build_ternary_op except this line
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .add_owned_output(maybe_get_output())
    .add_owned_const_input(self)
    .add_owned_const_input(a)
    .add_owned_const_input(b));
}

} // namespace at::meta


namespace at::native {

DEFINE_DISPATCH(betainc_stub);

TORCH_IMPL_FUNC(special_betainc_out) (const Tensor& self, const Tensor& a, const Tensor& b, const Tensor& result) {
  betainc_stub(device_type(), *this);
}

static Tensor _log_gamma_correction(const Tensor& x) {
  static const std::vector<double> const_vec = {
        0.833333333333333e-01L,
        -0.277777777760991e-02L,
        0.793650666825390e-03L,
        -0.595202931351870e-03L,
        0.837308034031215e-03L,
        -0.165322962780713e-02L,
        };
  Tensor minimax_coeff = at::tensor(const_vec, at::TensorOptions().dtype(x.dtype()).device(x.device()));

  Tensor inverse_x = at::reciprocal(x);
  Tensor inverse_x_squared = inverse_x * inverse_x;
  Tensor accum = minimax_coeff[5];
  for (int i = 4; i >= 0; i--) {
    accum = accum * inverse_x_squared + minimax_coeff[i];
  }
  return accum * inverse_x;
}

static Tensor _log_gamma_difference_big_y(const Tensor& x, const Tensor& y) {
  at::ScalarType dtype = at::promoteTypes(x.scalar_type(), y.scalar_type());
  auto options = at::TensorOptions().dtype(dtype).device(x.device());
  const Tensor half = at::scalar_tensor(0.5, options);
  const Tensor one = at::scalar_tensor(1.0, options);

  Tensor cancelled_stirling = (-one * (x + y - half) * at::log1p(x / y)
                               - x * at::log(y) + x);

  Tensor correction = _log_gamma_correction(y) - _log_gamma_correction(x + y);
  return correction + cancelled_stirling;
}

Tensor special_betaln(const Tensor& x, const Tensor& y) {
  TORCH_CHECK(
      x.device() == y.device(),
      "the paramters of betaln must be on the same device.");

  //dtype
  at::ScalarType dtype = at::promoteTypes(x.scalar_type(), y.scalar_type());
  auto options = at::TensorOptions().dtype(dtype).device(x.device());

  Tensor _x = at::minimum(x, y);
  Tensor _y = at::maximum(x, y);
  const Tensor half = at::scalar_tensor(0.5, options);
  const Tensor two = at::scalar_tensor(2.0, options);
  const Tensor eight = at::scalar_tensor(8.0, options);

  Tensor log2pi = at::log(two * at::scalar_tensor(c10::pi<double>, options));

  // Two large arguments case: _y >= _x >= 8
  Tensor log_beta_two_large = (half * log2pi
                               - half * at::log(_y)
                               + _log_gamma_correction(_x)
                               + _log_gamma_correction(_y)
                               - _log_gamma_correction(_x + _y)
                               + (_x - half) * at::log(_x / (_x + _y))
                               - _y * at::log1p(_x / _y));
  // Small arguments case: _x < 8, _y >= 8.
  Tensor log_beta_one_large = at::lgamma(_x) + _log_gamma_difference_big_y(_x, _y);

  // Small arguments case: _x <= _y < 8.
  Tensor log_beta_small = at::lgamma(_x) + at::lgamma(_y) - at::lgamma(_x + _y);

  return at::where(_x >= eight,
                   log_beta_two_large,
                   at::where(_y >= eight,
                             log_beta_one_large,
                             log_beta_small));
}

Tensor& special_betaln_out(const Tensor& a, const Tensor& b, Tensor& result) {
  TORCH_CHECK(
      a.device() == result.device(),
      "the paramters of betaln_out must be on the same device.");

  Tensor result_tmp = at::special_betaln(a, b);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor special_betaln(const Tensor& a, const Scalar& b) {
  return at::special_betaln(a, wrapped_scalar_tensor(b, a.device()));
}

Tensor special_betaln(const Scalar& a, const Tensor& b) {
  return at::special_betaln(wrapped_scalar_tensor(a, b.device()), b);
}

Tensor& special_betaln_out(const Tensor& a, const Scalar& b, Tensor& result) {
  return at::special_betaln_out(result, a, wrapped_scalar_tensor(b, a.device()));
}

Tensor& special_betaln_out(const Scalar& a, const Tensor& b, Tensor& result) {
  return at::special_betaln_out(result, wrapped_scalar_tensor(a, b.device()), b);
}

Tensor special_betainc(const Tensor& self, const Scalar& a, const Scalar& b) {
  return at::special_betainc(self, wrapped_scalar_tensor(a), wrapped_scalar_tensor(b));
}

Tensor special_betainc(const Tensor& self, const Scalar& a, const Tensor& b) {
  return at::special_betainc(self, wrapped_scalar_tensor(a), b);
}

Tensor special_betainc(const Tensor& self, const Tensor& a, const Scalar& b) {
  return at::special_betainc(self, a, wrapped_scalar_tensor(b));
}

Tensor special_betainc(const Scalar& self, const Tensor& a, const Tensor& b) {
  return at::special_betainc(wrapped_scalar_tensor(self), a, b);
}

Tensor& special_betainc_out(const Tensor& self, const Scalar& a, const Scalar& b, Tensor& result) {
  return at::special_betainc_out(result, self, wrapped_scalar_tensor(a), wrapped_scalar_tensor(b));
}

Tensor& special_betainc_out(const Tensor& self, const Scalar& a, const Tensor& b, Tensor& result) {
  return at::special_betainc_out(result, self, wrapped_scalar_tensor(a), b);
}

Tensor& special_betainc_out(const Tensor& self, const Tensor& a, const Scalar& b, Tensor& result) {
  return at::special_betainc_out(result, self, a, wrapped_scalar_tensor(b));
}

Tensor& special_betainc_out(const Scalar& self, const Tensor& a, const Tensor& b, Tensor& result) {
  return at::special_betainc_out(result, wrapped_scalar_tensor(self), a, b);
}

} // namespace at::native
