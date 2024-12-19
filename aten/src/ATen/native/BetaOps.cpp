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
#include <ATen/ops/special_betaincinv.h>
#include <ATen/ops/special_betaincinv_native.h>
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

static Tensor _betaincinv_initial_approx(const Tensor& y, const Tensor& a, const Tensor& b, at::ScalarType& dtype) {
    /* Computes an initial approximation for `betaincinv(y, a, b)`. */
  std::tuple<Tensor, Tensor, Tensor> eps_tiny_maxexp = AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "_betaincinv_eps_tiny_maxexp",
        [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor eps = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon(), y.device());
    Tensor tiny = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(), y.device()); //min == lowest, tiny == min
    Tensor maxexp = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max_exponent, y.device());
    return std::make_tuple(std::move(eps), std::move(tiny), std::move(maxexp));
  });
  const Tensor eps = std::move(std::get<0>(eps_tiny_maxexp));
  const Tensor tiny = std::move(std::get<1>(eps_tiny_maxexp));
  const Tensor maxexp = std::move(std::get<2>(eps_tiny_maxexp));
  auto options = at::TensorOptions().dtype(dtype).device(y.device());
  const Tensor one = at::scalar_tensor(1.0, options);
  const Tensor two = at::scalar_tensor(2.0, options);
  const Tensor three = at::scalar_tensor(3.0, options);
  const Tensor five = at::scalar_tensor(5.0, options);
  const Tensor six = at::scalar_tensor(6.0, options);
  const Tensor max_log = (maxexp - one) * at::log(two);

  // When min(a, b) >= 1, we use the approximation proposed by [1].

  // Equation 26.5.22 [1, page 945].
  Tensor yp = - at::special_ndtri(y);
  Tensor inv_2a_minus_one = at::reciprocal(two * a - one);
  Tensor inv_2b_minus_one = at::reciprocal(two * b - one);
  Tensor lmb = (at::square(yp) - three) / six;
  Tensor h = two * at::reciprocal(inv_2a_minus_one + inv_2b_minus_one);
  Tensor w = (yp * at::sqrt(h + lmb) / h -
          (inv_2b_minus_one - inv_2a_minus_one) *
          (lmb + five / six - two / (three * h)));
  Tensor result_for_large_a_and_b = a / (a + b * at::exp(two * w));

  /* When min(a, b) < 1 and max(a, b) >= 1, we use the approximation proposed by
   * [2]. This approximation depends on the following approximation for betainc:
   *   betainc(x, a, b) ~=
   *       x ** a / (integral_approx * a) , when x <= mean ,
   *       (1 - x) ** b / (integral_approx * b) , when x > mean ,
   * where:
   *   integral_approx = (mean ** a) / a + (mean_complement ** b) / b ,
   *   mean = a / (a + b) ,
   *   mean_complement = 1 - mean = b / (a + b) .
   *   We invert betainc(x, a, b) with respect to x in the proper regime. */

  // Equation 6.4.7 [2, page 271]
  Tensor a_plus_b = a + b;
  Tensor mean = a / a_plus_b;
  Tensor mean_complement = b / a_plus_b;
  Tensor integral_approx_part_a = at::exp(at::xlogy(a, mean) - at::log(a));
  Tensor integral_approx_part_b = at::exp(at::xlogy(b, mean_complement) -
                                      at::log(b));
  Tensor integral_approx = integral_approx_part_a + integral_approx_part_b;

  // Solve Equation 6.4.8 [2, page 271] for x in the respective regimes.
  Tensor inv_a = at::reciprocal(a);
  Tensor inv_b = at::reciprocal(b);
  Tensor result_for_small_a_or_b = at::where(
      y <= (integral_approx_part_a / integral_approx),
          at::exp(at::xlogy(inv_a, y) + at::xlogy(inv_a, a) +
                  at::xlogy(inv_a, integral_approx)),
          -at::expm1(at::special_xlog1py(inv_b, -y) + at::xlogy(inv_b, b) +
                     at::xlogy(inv_b, integral_approx)));

  /* And when max(a, b) < 1, we use the approximation proposed by [3] for the
   * same domain:
   *   betaincinv(y, a, b) ~= xg / (1 + xg) ,
   * where:
   *   xg = (a * y * Beta(a, b)) ** (1 / a) . */
  Tensor log_xg = at::xlogy(inv_a, a) + at::xlogy(inv_a, y) + (
      inv_a * at::special_betaln(a, b));
  Tensor xg = at::exp(at::minimum(log_xg, max_log));
  Tensor result_for_small_a_and_b = xg / (one + xg);

  Tensor result = at::where(
      at::minimum(a, b) >= one,
          result_for_large_a_and_b,
          at::where(at::maximum(a, b) < one,
              result_for_small_a_and_b,
              result_for_small_a_or_b));

  return at::clamp(result, tiny, one - eps);
}

static Tensor _betaincinv_computation(const Tensor& y, const Tensor& a, const Tensor& b) {
  at::ScalarType dtype_orig = at::promoteTypes(at::promoteTypes(a.scalar_type(), b.scalar_type()), y.scalar_type());
  bool should_promote_dtype = ((dtype_orig == at::ScalarType::BFloat16) | (dtype_orig == at::ScalarType::Half)) ? true : false;
  at::ScalarType dtype = should_promote_dtype ? at::ScalarType::Float : dtype_orig;
  Tensor _y = y.to(dtype_orig);
  Tensor _a = a.to(dtype_orig);
  Tensor _b = b.to(dtype_orig);

  if (should_promote_dtype) {
    _y = _y.to(dtype);
    _a = _a.to(dtype);
    _b = _b.to(dtype);
  }

  std::tuple<Tensor, Tensor, Tensor> eps_tiny_nan = AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "_betaincinv_computation_eps_tiny",
        [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor eps = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon(), _y.device());
    Tensor tiny = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(), _y.device()); //min == lowest, tiny == min
    Tensor nan = at::scalar_to_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::quiet_NaN(), _y.device()); 
    return std::make_tuple(std::move(eps), std::move(tiny), std::move(nan));
  });
  auto options = at::TensorOptions().dtype(dtype).device(_y.device());
  const Tensor eps = std::move(std::get<0>(eps_tiny_nan));
  const Tensor tiny = std::move(std::get<1>(eps_tiny_nan));
  const Tensor nan = std::move(std::get<2>(eps_tiny_nan));
  const Tensor zero = at::scalar_tensor(0.0, options);
  const Tensor half = at::scalar_tensor(0.5, options);
  const Tensor one = at::scalar_tensor(1.0, options);
  const Tensor two = at::scalar_tensor(2.0, options);
  const Tensor halley_correction_min = at::scalar_tensor(0.5, options);
  const Tensor halley_correction_max = at::scalar_tensor(1.5, options);
  const Tensor _true = at::scalar_tensor(true, options);

  /* When betainc(0.5, a, b) < y, we apply the symmetry relation given
   * here: https://dlmf.nist.gov/8.17.E4
   *   torch.special.betainc(x, a, b) = 1 - torch.special.betainc(1 - x, b, a) .
   * If dtype is float32, we have additional conditions to apply this relation:
   *   (a < 1) & (b < 1) & (torch_special.betainc(a / (a + b), a, b) < y) . */
  Tensor a_and_b_are_small;
  Tensor error_at_mean;

  Tensor error_at_half = at::special_betainc(half, _a, _b);
  Tensor use_symmetry_relation;
  if (dtype == at::ScalarType::Float) {
      a_and_b_are_small = (_a < one) & (_b < one);
      error_at_mean = at::special_betainc(_a / (_a + _b), _a, _b) - _y;
      use_symmetry_relation = (error_at_half < zero) & a_and_b_are_small & (
          error_at_mean < zero);
  } else { // T is double
      use_symmetry_relation = (error_at_half < zero);
  }

  Tensor a_orig = _a, y_orig = _y;
  _a = at::where(use_symmetry_relation, _b, _a);
  _b = at::where(use_symmetry_relation, a_orig, _b);
  _y = at::where(use_symmetry_relation, one - _y, _y);

  Tensor a_minus_1 = _a - one;
  Tensor b_minus_1 = _b - one;
  Tensor lbeta_a_and_b = at::special_betaln(_a, _b);
  Tensor two_tiny = two * tiny;

  // max_iterations was taken from [4] and tolerance was set by experimentation.
  int max_iterations = 0;
  Tensor tolerance;
  if (dtype == at::ScalarType::Float) {
      max_iterations = 10;
      tolerance = at::scalar_tensor(8.0, options) * eps;
  } else { // T is double
      max_iterations = 8;
      tolerance = at::scalar_tensor(4096.0, options) * eps;
  }

  Tensor initial_candidate = _betaincinv_initial_approx(_y, _a, _b, dtype);
  // Bracket the solution with the interval (low, high).
  Tensor initial_low = at::zeros_like(_y);
  Tensor initial_high;
  if (dtype == at::ScalarType::Float) {
      initial_high = at::ones_like(_y) * at::where(
          a_and_b_are_small & (error_at_mean < zero), half, one);
  } else {
      initial_high = at::ones_like(_y) * half;
  }

  Tensor should_stop = (_y == initial_low) | (_y == initial_high);
  Tensor low = initial_low;
  Tensor high = initial_high;
  Tensor candidate = initial_candidate;

  // root_finding_iteration
  for (int i = 0; i < max_iterations; i++) {
      if (should_stop.all().equal(_true))
          break;
      Tensor error = at::special_betainc(candidate, _a, _b) - _y;
      Tensor error_over_der = error / at::exp(
          at::xlogy(a_minus_1, candidate) +
          at::special_xlog1py(b_minus_1, -candidate) -
          lbeta_a_and_b);
      Tensor second_der_over_der = a_minus_1 / candidate - b_minus_1 / (one - candidate);

      /* Following [2, section 9.4.2, page 463], we limit the influence of the
         Halley's correction to the Newton's method, since this correction can
         reduce the Newton's region of convergence. We set minimum and maximum
         values for this correction by experimentation. */
      Tensor halley_correction = at::clamp(one - half * error_over_der * second_der_over_der,
                                           halley_correction_min, halley_correction_max);
      Tensor halley_delta = error_over_der / halley_correction;
      Tensor halley_candidate = at::where(should_stop, candidate, candidate - halley_delta);

      /* Fall back to bisection if the current step would take the new candidate
       * out of bounds. */
      Tensor new_candidate = at::where(
          halley_candidate <= low,
              half * (candidate + low),
              at::where(halley_candidate >= high,
                  half * (candidate + high),
                  halley_candidate));

      Tensor new_delta = candidate - new_candidate;
      Tensor new_delta_is_negative = (new_delta < zero);
      Tensor new_low = at::where(new_delta_is_negative, candidate, low);
      Tensor new_high = at::where(new_delta_is_negative, high, candidate);

      Tensor adjusted_tolerance = at::maximum(tolerance * new_candidate, two_tiny);
      should_stop = (should_stop | (at::abs(new_delta) < adjusted_tolerance) |
                     at::eq(new_low, new_high));
      low = std::move(new_low);
      high = std::move(new_high);
      candidate = std::move(new_candidate);
  }

  Tensor result = std::move(candidate);

  // If we are taking advantage of the symmetry relation, we have to adjust the
  // input y and the solution.
  _y = y_orig;
  result = at::where(use_symmetry_relation, one - at::maximum(result, eps), result);

  // Handle trivial cases.
  result = at::where((at::eq(_y, zero) | at::eq(_y, one)), _y, result);

  // Determine if the inputs are out of range (should return NaN output).
  Tensor result_is_nan = (_a <= zero) | (_b <= zero) | (_y < zero) | (_y > one);
  result = at::where(result_is_nan, nan, result);

  if (should_promote_dtype)
    result = result.to(dtype_orig);

  return result;
}

Tensor special_betaincinv(const Tensor& self, const Tensor& a, const Tensor& b) {
  TORCH_CHECK(
      self.device() == a.device() && a.device() == b.device(),
      "the paramters of betaincinv must be on the same device.");
  return _betaincinv_computation(self, a, b);
}

Tensor& special_betaincinv_out(const Tensor& self, const Tensor& a, const Tensor& b, Tensor& result) {
  TORCH_CHECK(
      self.device() == result.device(),
      "the paramters of betaincinv_out must be on the same device.");

  Tensor result_tmp = at::special_betaincinv(self, a, b);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor special_betaincinv(const Tensor& self, const Scalar& a, const Scalar& b) {
  return at::special_betaincinv(self, wrapped_scalar_tensor(a, self.device()), wrapped_scalar_tensor(b, self.device()));
}

Tensor special_betaincinv(const Tensor& self, const Scalar& a, const Tensor& b) {
  return at::special_betaincinv(self, wrapped_scalar_tensor(a, self.device()), b);
}

Tensor special_betaincinv(const Tensor& self, const Tensor& a, const Scalar& b) {
  return at::special_betaincinv(self, a, wrapped_scalar_tensor(b, self.device()));
}

Tensor special_betaincinv(const Scalar& self, const Tensor& a, const Tensor& b) {
  return at::special_betaincinv(wrapped_scalar_tensor(self, a.device()), a, b);
}

Tensor& special_betaincinv_out(const Tensor& self, const Scalar& a, const Scalar& b, Tensor& result) {
  return at::special_betaincinv_out(result, self, wrapped_scalar_tensor(a, self.device()), wrapped_scalar_tensor(b, self.device()));
}

Tensor& special_betaincinv_out(const Tensor& self, const Scalar& a, const Tensor& b, Tensor& result) {
  return at::special_betaincinv_out(result, self, wrapped_scalar_tensor(a, self.device()), b);
}

Tensor& special_betaincinv_out(const Tensor& self, const Tensor& a, const Scalar& b, Tensor& result) {
  return at::special_betaincinv_out(result, self, a, wrapped_scalar_tensor(b, self.device()));
}

Tensor& special_betaincinv_out(const Scalar& self, const Tensor& a, const Tensor& b, Tensor& result) {
  return at::special_betaincinv_out(result, wrapped_scalar_tensor(self, a.device()), a, b);
}

} // namespace at::native
