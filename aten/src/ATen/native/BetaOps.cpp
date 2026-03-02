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

#include <ATen/ops/cat.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/special_digamma.h>
#include <ATen/ops/special_digamma_native.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/digamma_native.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unbind_native.h>


#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scalar_tensor_native.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/broadcast_tensors.h>
#include <ATen/ops/broadcast_tensors_native.h>

#endif

namespace at::meta {

#define FLOAT_OP_CONFIG()                       \
  TensorIteratorConfig()                        \
    .set_check_mem_overlap(true)                \
    .allow_cpu_scalars(true)                    \
    .promote_inputs_to_common_dtype(true)       \
    .cast_common_dtype_to_outputs(true)         \
    .enforce_safe_casting_to_output(true)       \
    .promote_integer_inputs_to_float(true)

TORCH_META_FUNC(special_betainc) (const Tensor& a, const Tensor& b, const Tensor& x) {
  build(FLOAT_OP_CONFIG()
      .add_output(maybe_get_output())
      .add_const_input(a)
      .add_const_input(b)
      .add_const_input(x));
}

TORCH_META_FUNC(special_betaln) (const Tensor& a, const Tensor& b) {
  build(FLOAT_OP_CONFIG()
      .add_output(maybe_get_output())
      .add_const_input(a)
      .add_const_input(b));
}

TORCH_META_FUNC(special_betaincinv) (const Tensor& a, const Tensor& b, const Tensor& y) {
  build(FLOAT_OP_CONFIG()
      .add_output(maybe_get_output())
      .add_const_input(a)
      .add_const_input(b)
      .add_const_input(y));
}

} // namespace at::meta


namespace at::native {

DEFINE_DISPATCH(betainc_stub);

TORCH_IMPL_FUNC(special_betainc_out) (const Tensor& a, const Tensor& b, const Tensor& x, const Tensor& result) {
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
  Tensor cancelled_stirling = (-1.0 * (x + y - 0.5) * at::log1p(x / y)
                               - x * at::log(y) + x);
  Tensor correction = _log_gamma_correction(y) - _log_gamma_correction(x + y);
  return correction + cancelled_stirling;
}

static Tensor _special_betaln(const Tensor& x, const Tensor& y) {
  at::ScalarType dtype_orig = at::promoteTypes(x.scalar_type(), y.scalar_type());
  bool should_promote_dtype = ((dtype_orig == at::ScalarType::BFloat16) | (dtype_orig == at::ScalarType::Half)) ? true : false;
  at::ScalarType dtype = should_promote_dtype ? at::ScalarType::Float : dtype_orig;
  Tensor __x = x.to(dtype);
  Tensor __y = y.to(dtype);

  auto options = at::TensorOptions().dtype(dtype).device(x.device());

  Tensor _x = at::minimum(__x, __y);
  Tensor _y = at::maximum(__x, __y);
  Tensor log2pi = at::log(2.0 * at::scalar_tensor(c10::pi<double>, options));

  // Two large arguments case: _y >= _x >= 8
  Tensor log_beta_two_large = (0.5 * log2pi
                               - 0.5 * at::log(_y)
                               + _log_gamma_correction(_x)
                               + _log_gamma_correction(_y)
                               - _log_gamma_correction(_x + _y)
                               + (_x - 0.5) * at::log(_x / (_x + _y))
                               - _y * at::log1p(_x / _y));
  // Small arguments case: _x < 8, _y >= 8.
  Tensor log_beta_one_large = at::lgamma(_x) + _log_gamma_difference_big_y(_x, _y);

  // Small arguments case: _x <= _y < 8.
  Tensor log_beta_small = at::lgamma(_x) + at::lgamma(_y) - at::lgamma(_x + _y);

  Tensor result = at::where(_x >= 8.0,
                   log_beta_two_large,
                   at::where(_y >= 8.0,
                             log_beta_one_large,
                             log_beta_small));
  if (should_promote_dtype)
    result = result.to(dtype_orig);

  return result;
}

TORCH_IMPL_FUNC(special_betaln_out) (const Tensor& a, const Tensor& b, const Tensor& result) {
  TORCH_CHECK(!a.is_complex() && !b.is_complex(),
              "special.betaln is not yet implemented for complex tensors.");

  TORCH_CHECK(!isIntegralType(a.scalar_type(), true) || !isIntegralType(b.scalar_type(), true),
              "special.betaln must have at least one floating parameter.");
  const Tensor&& result_tmp = _special_betaln(a, b);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
}

static Tensor _betaincinv_initial_approx(const Tensor& a, const Tensor& b, const Tensor& y, at::ScalarType& dtype) {
    /* Computes an initial approximation for `betaincinv(a, b, y)`. */
  const auto&& [eps, tiny, maxexp] = AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "_betaincinv_eps_tiny_maxexp",
        [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor eps = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon(), y.options());
    Tensor tiny = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(), y.options()); //min == lowest, tiny == min
    Tensor maxexp = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max_exponent, y.options());
    return std::make_tuple(std::move(eps), std::move(tiny), std::move(maxexp));
  });
  auto options = at::TensorOptions().dtype(dtype).device(y.device());
  const Tensor two = at::scalar_tensor(2.0, options);
  const Tensor six = at::scalar_tensor(6.0, options);
  const Tensor max_log = (maxexp - 1.0) * at::log(two);

  // When min(a, b) >= 1, we use the approximation proposed by [1].

  // Equation 26.5.22 [1, page 945].
  Tensor yp = - at::special_ndtri(y);
  Tensor inv_2a_minus_one = at::reciprocal(2.0 * a - 1.0);
  Tensor inv_2b_minus_one = at::reciprocal(2.0 * b - 1.0);
  Tensor lmb = (at::square(yp) - 3.0) / 6.0;
  Tensor h = 2.0 * at::reciprocal(inv_2a_minus_one + inv_2b_minus_one);
  Tensor w = (yp * at::sqrt(h + lmb) / h -
          (inv_2b_minus_one - inv_2a_minus_one) *
          (lmb + 5.0 / six - 2.0 / (3.0 * h)));
  Tensor result_for_large_a_and_b = a / (a + b * at::exp(2.0 * w));

  /* When min(a, b) < 1 and max(a, b) >= 1, we use the approximation proposed by
   * [2]. This approximation depends on the following approximation for betainc:
   *   betainc(a, b, x) ~=
   *       x ** a / (integral_approx * a) , when x <= mean ,
   *       (1 - x) ** b / (integral_approx * b) , when x > mean ,
   * where:
   *   integral_approx = (mean ** a) / a + (mean_complement ** b) / b ,
   *   mean = a / (a + b) ,
   *   mean_complement = 1 - mean = b / (a + b) .
   *   We invert betainc(a, b, x) with respect to x in the proper regime. */

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
   *   betaincinv(a, b, y) ~= xg / (1 + xg) ,
   * where:
   *   xg = (a * y * Beta(a, b)) ** (1 / a) . */
  Tensor log_xg = at::xlogy(inv_a, a) + at::xlogy(inv_a, y) + (
      inv_a * at::special_betaln(a, b));
  Tensor xg = at::exp(at::minimum(log_xg, max_log));
  Tensor result_for_small_a_and_b = xg / (1.0 + xg);

  Tensor result = at::where(
      at::minimum(a, b) >= 1.0,
          result_for_large_a_and_b,
          at::where(at::maximum(a, b) < 1.0,
              result_for_small_a_and_b,
              result_for_small_a_or_b));

  return at::clamp(result, tiny, 1.0 - eps);
}

static Tensor _betaincinv_computation(const Tensor& a, const Tensor& b, const Tensor& y) {
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

  const auto&& [eps, tiny, nan] = AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "_betaincinv_computation_eps_tiny",
        [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor eps = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon(), _y.options());
    Tensor tiny = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(), _y.options()); //min == lowest, tiny == min
    Tensor nan = at::scalar_tensor(std::numeric_limits<at::scalar_value_type<scalar_t>::type>::quiet_NaN(), _y.options());
    return std::make_tuple(std::move(eps), std::move(tiny), std::move(nan));
  });
  auto options = at::TensorOptions().dtype(dtype).device(_y.device());
  const Tensor half = at::scalar_tensor(0.5, options);
  const Tensor one = at::scalar_tensor(1.0, options);
  const Tensor halley_correction_min = at::scalar_tensor(0.5, options);
  const Tensor halley_correction_max = at::scalar_tensor(1.5, options);
  const Tensor _true = at::scalar_tensor(true, _y.device());

  /* When betainc(0.5, a, b) < y, we apply the symmetry relation given
   * here: https://dlmf.nist.gov/8.17.E4
   *   torch.special.betainc(a, b, x) = 1 - torch.special.betainc(b, a, 1 - x) .
   * If dtype is float32, we have additional conditions to apply this relation:
   *   (a < 1) & (b < 1) & (torch_special.betainc(a / (a + b), a, b) < y) . */
  Tensor a_and_b_are_small;
  Tensor error_at_mean;

  Tensor error_at_half = at::special_betainc(_a, _b, half) - _y;
  Tensor use_symmetry_relation;
  if (dtype == at::ScalarType::Float) {
      a_and_b_are_small = (_a < 1.0) & (_b < 1.0);
      error_at_mean = at::special_betainc(_a, _b, _a / (_a + _b)) - _y;
      use_symmetry_relation = (error_at_half < 0.0) & a_and_b_are_small & (
          error_at_mean < 0.0);
  } else { // T is double
      use_symmetry_relation = (error_at_half < 0.0);
  }

  Tensor a_orig = _a, y_orig = _y;
  _a = at::where(use_symmetry_relation, _b, _a);
  _b = at::where(use_symmetry_relation, a_orig, _b);
  _y = at::where(use_symmetry_relation, 1.0 - _y, _y);

  Tensor a_minus_1 = _a - 1.0;
  Tensor b_minus_1 = _b - 1.0;
  Tensor lbeta_a_and_b = at::special_betaln(_a, _b);
  Tensor two_tiny = 2.0 * tiny;

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

  Tensor initial_candidate = _betaincinv_initial_approx(_a, _b, _y, dtype);
  // Bracket the solution with the interval (low, high).
  Tensor initial_low = at::zeros_like(_y);
  Tensor initial_high;
  if (dtype == at::ScalarType::Float) {
      initial_high = at::ones_like(_y) * at::where(
          a_and_b_are_small & (error_at_mean < 0.0), 0.5, 1.0);
  } else {
      initial_high = at::ones_like(_y) * 0.5;
  }

  Tensor should_stop = (_y == initial_low) | (_y == initial_high);
  Tensor low = initial_low;
  Tensor high = initial_high;
  Tensor candidate = initial_candidate;

  // root_finding_iteration
  for (int i = 0; i < max_iterations; i++) {
      if (should_stop.all().equal(_true))
          break;
      Tensor error = at::special_betainc(_a, _b, candidate) - _y;
      Tensor error_over_der = error / at::exp(
          at::xlogy(a_minus_1, candidate) +
          at::special_xlog1py(b_minus_1, -candidate) -
          lbeta_a_and_b);
      Tensor second_der_over_der = a_minus_1 / candidate - b_minus_1 / (1.0 - candidate);

      /* Following [2, section 9.4.2, page 463], we limit the influence of the
         Halley's correction to the Newton's method, since this correction can
         reduce the Newton's region of convergence. We set minimum and maximum
         values for this correction by experimentation. */
      Tensor halley_correction = at::clamp(1.0 - 0.5 * error_over_der * second_der_over_der,
                                           halley_correction_min, halley_correction_max);
      Tensor halley_delta = error_over_der / halley_correction;
      Tensor halley_candidate = at::where(should_stop, candidate, candidate - halley_delta);

      /* Fall back to bisection if the current step would take the new candidate
       * out of bounds. */
      Tensor new_candidate = at::where(
          halley_candidate <= low,
              0.5 * (candidate + low),
              at::where(halley_candidate >= high,
                  0.5 * (candidate + high),
                  halley_candidate));

      Tensor new_delta = candidate - new_candidate;
      Tensor new_delta_is_negative = (new_delta < 0.0);
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
  result = at::where(use_symmetry_relation, 1.0 - at::maximum(result, eps), result);

  // Handle trivial cases.
  result = at::where((at::eq(_y, 0.0) | at::eq(_y, one)), _y, result);

  // Determine if the inputs are out of range (should return NaN output).
  Tensor result_is_nan = (_a <= 0.0) | (_b <= 0.0) | (_y < 0.0) | (_y > 1.0);
  result = at::where(result_is_nan, nan, result);

  if (should_promote_dtype)
    result = result.to(dtype_orig);

  return result;
}

TORCH_IMPL_FUNC(special_betaincinv_out) (const Tensor& a, const Tensor& b, const Tensor& y, const Tensor& result) {
  const Tensor&& result_tmp = _betaincinv_computation(a, b, y);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
}


static inline std::tuple<Tensor, Tensor> _betainc_even_partial_numerator(
    const int32_t iteration,
    const Tensor& a,
    const Tensor& b,
    const Tensor& x) {
  // Even partial numerator used in the continued fraction for betainc.
  /*
   * This function computes the partial numerator d_{2m} that is specified
   * here: https://dlmf.nist.gov/8.17.E23
   */
  auto options = at::TensorOptions().dtype(x.dtype()).device(x.device());
  const Tensor two = at::scalar_tensor(2.0, options);
  int32_t m = iteration;
  Tensor a_plus_2m = a + two * m;
  Tensor a_plus_2m_minus_one = a_plus_2m - 1.0;
  Tensor denominator = a_plus_2m * a_plus_2m_minus_one;

  Tensor db = m * x / denominator;
  Tensor value = db * (b - m);
  Tensor da = -value * (a_plus_2m + a_plus_2m_minus_one) / denominator;

  return std::make_tuple(
      std::move(value), at::cat({std::move(da), std::move(db)}, -1));
}

static inline std::tuple<Tensor, Tensor> _betainc_odd_partial_numerator(
    const int32_t iteration,
    const Tensor& a,
    const Tensor& b,
    const Tensor& x) {
  // Odd partial numerator used in the continued fraction for betainc.
  /*
   * This function computes the partial numerator d_{2m + 1} that is specified
   * here: https://dlmf.nist.gov/8.17.E23
   */
  int32_t m = iteration;
  Tensor a_plus_m = a + m;
  Tensor a_plus_2m = a_plus_m + m;
  Tensor a_plus_2m_plus_one = a_plus_2m + 1.0;
  Tensor a_plus_b_plus_m = a_plus_m + b;
  Tensor denominator = a_plus_2m * a_plus_2m_plus_one;

  Tensor db = -a_plus_m * x / denominator;
  Tensor value = db * a_plus_b_plus_m;
  Tensor da = -value * ((a_plus_2m + a_plus_2m_plus_one) / denominator) -
      x * (2.0 * a_plus_m + b) / denominator;

  return std::make_tuple(
      std::move(value), at::cat({std::move(da), std::move(db)}, -1));
}

static inline std::tuple<Tensor, Tensor, Tensor> _betainc_modified_lentz_method(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x,
    const Tensor& use_continued_fraction) {
  // Returns the continued fraction for betainc by modified Lentz's method.
  /*
   * This function implements the method described in the appendix of [1] for
   * evaluating continued fractions.
   * [1] Thompson, Ian J., and A. Ross Barnett.
   *     Coulomb and Bessel functions of complex arguments and order.
   *     Journal of Computational Physics 64.2 (1986): 490-509.
   *     https://www.fresco.org.uk/papers/Thompson-JCP64p490.pdf
   */
  // a, b, and x have same dtype
  auto [eps, tiny] = AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "__betainc_modified_lentz_method_eps_tiny",
      [&]() -> std::tuple<Tensor, Tensor> {
        Tensor eps = at::scalar_tensor(
            std::numeric_limits<
                at::scalar_value_type<scalar_t>::type>::epsilon(),
            x.options());
        Tensor tiny = at::scalar_tensor(
            std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(),
            x.options()); // min == lowest, tiny == min
        return std::make_tuple(std::move(eps), std::move(tiny));
      });

  const Tensor _true = at::scalar_tensor(true, x.device());

  // max_iterations and tolerance were taken from Cephes.
  int32_t max_iterations = 100; // at::kFloat
  Tensor tolerance = eps;

  if (x.scalar_type() == at::kDouble) {
    max_iterations = 300;
    tolerance *= 3.0;
  }
  Tensor small = at::sqrt(tiny);
  /* Assume all input Tensors have the same shape. The extra dimension is
   * needed to compute the gradients with respect to a and b. */
  const Tensor& _a = a.unsqueeze(-1);
  const Tensor& _b = b.unsqueeze(-1);
  const Tensor& _x = x.unsqueeze(-1);
  const Tensor& _use_continued_fraction = use_continued_fraction.unsqueeze(-1);

  auto __continued_fraction_step =
      [&](int32_t iteration,
          const std::vector<Tensor>& values,
          const std::vector<Tensor>& gradients,
          const std::function<std::tuple<Tensor, Tensor>(
              const int32_t, const Tensor&, const Tensor&, const Tensor&)>&
              partial_numerator_fn)
      -> std::tuple<std::vector<Tensor>, std::vector<Tensor>, Tensor> {
    const Tensor& ratio_numerators = values.at(0);
    const Tensor& ratio_denominators = values.at(1);
    const Tensor& convergent = values.at(2);
    const Tensor& dratio_numerators = gradients.at(0);
    const Tensor& dratio_denominators = gradients.at(1);
    const Tensor& dconvergent = gradients.at(2);
    const auto&& [partial_numerator, dpartial_numerator] =
        partial_numerator_fn(iteration, _a, _b, _x);

    // new_ratio_numerators = C_n = A_n / A_{n - 1}
    Tensor new_ratio_numerators = 1.0 + partial_numerator / ratio_numerators;
    new_ratio_numerators = at::where(
        at::abs(new_ratio_numerators) < small, small, new_ratio_numerators);

    // new_ratio_denominators = D_n = B_{n - 1} / B_n
    Tensor new_ratio_denominators =
        1.0 + partial_numerator * ratio_denominators;
    new_ratio_denominators = at::where(
        at::abs(new_ratio_denominators) < small, small, new_ratio_denominators);
    new_ratio_denominators = at::reciprocal(new_ratio_denominators);

    // new_convergent = h_n = A_n / B_n = h_{n - 1} * C_n * D_n;
    Tensor delta = new_ratio_numerators * new_ratio_denominators;
    Tensor new_convergent = convergent * delta;

    Tensor new_dratio_numerators =
        (dpartial_numerator * ratio_numerators -
         partial_numerator * dratio_numerators);
    new_dratio_numerators =
        new_dratio_numerators / at::square(ratio_numerators);
    Tensor new_dratio_denominators =
        (dpartial_numerator * ratio_denominators +
         partial_numerator * dratio_denominators);
    new_dratio_denominators =
        -new_dratio_denominators * at::square(new_ratio_denominators);

    Tensor new_dconvergent = dconvergent * delta +
        (convergent * new_dratio_numerators * new_ratio_denominators);
    new_dconvergent = new_dconvergent +
        (convergent * new_dratio_denominators * new_ratio_numerators);

    std::vector<Tensor> new_values = {
        std::move(new_ratio_numerators),
        std::move(new_ratio_denominators),
        std::move(new_convergent)};
    std::vector<Tensor> new_gradients = {
        std::move(new_dratio_numerators),
        std::move(new_dratio_denominators),
        std::move(new_dconvergent)};

    return std::make_tuple(
        std::move(new_values), std::move(new_gradients), std::move(delta));
  };

  auto __continued_fraction_evaluation =
      [&](const Tensor& should_stop,
          int32_t iteration,
          const std::vector<Tensor>& values,
          const std::vector<Tensor>& gradients)
      -> std::tuple<Tensor, int32_t, std::vector<Tensor>, std::vector<Tensor>> {
    // We run two steps of modified Lentz's method per iteration.
    // First step of the iteration: the even one.
    auto [_new_values, _new_gradients, _delta] = __continued_fraction_step(
        iteration, values, gradients, _betainc_even_partial_numerator);

    // Second step of the iteration: the odd one.
    auto [new_values, new_gradients, delta] = __continued_fraction_step(
        iteration, _new_values, _new_gradients, _betainc_odd_partial_numerator);
    Tensor stop = should_stop | (at::abs(delta - 1.0) < tolerance);
    return std::make_tuple(
        std::move(stop),
        iteration + 1,
        std::move(new_values),
        std::move(new_gradients));
  };

  Tensor apb = _a + _b;
  Tensor ap1 = _a + 1.0;
  // Initialization and first step of modified Lentz's method.
  Tensor initial_ratio_numerators = at::ones_like(_x);
  Tensor initial_ratio_denominators = 1.0 - apb * _x / ap1;
  initial_ratio_denominators = at::where(
      at::abs(initial_ratio_denominators) < small,
      small,
      initial_ratio_denominators);
  initial_ratio_denominators = at::reciprocal(initial_ratio_denominators);
  Tensor initial_convergent = initial_ratio_denominators;
  std::vector<Tensor> values = {
      std::move(initial_ratio_numerators),
      std::move(initial_ratio_denominators),
      std::move(initial_convergent)};

  Tensor initial_dratio_denominators =
      at::cat({1.0 - _b, ap1}, -1) * _x / at::square(_x * apb - ap1);
  Tensor initial_dratio_numerators =
      at::zeros_like(initial_dratio_denominators);
  Tensor initial_dconvergent = initial_dratio_denominators;
  std::vector<Tensor> gradients = {
      std::move(initial_dratio_numerators),
      std::move(initial_dratio_denominators),
      std::move(initial_dconvergent)};

  Tensor stop = ~_use_continued_fraction;

  for (int32_t i = 0; i < max_iterations; i++) {
    std::tuple<Tensor, int32_t, std::vector<Tensor>, std::vector<Tensor>> ret =
        __continued_fraction_evaluation(stop, i + 1, values, gradients);
    stop = std::get<0>(ret);
    values = std::get<2>(ret);
    gradients = std::get<3>(ret);
    if (stop.all().equal(_true)) // TODO: It can be bottleneck..
      break;
  }

  // Remove the previously added extra dimension: it is no longer needed.
  Tensor convergent = values.back().squeeze(-1);
  std::vector<Tensor> convergent_grads = at::unbind(gradients.back(), -1);

  return std::make_tuple(
      std::move(convergent),
      std::move(convergent_grads.at(0)),
      std::move(convergent_grads.at(1)));
}

static inline std::tuple<Tensor, Tensor> _betainc_der_continued_fraction(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x,
    const Tensor& use_continued_fraction) {
  // Returns the partial derivatives of betainc with respect to a and b.
  /*
   * This function evaluates betainc(a, b, x) by its continued fraction
   * expansion given here: https://dlmf.nist.gov/8.17.E22
   * We apply this function when the input (a, b, x) does not belong to the
   * proper region of computation of `_betainc_der_power_series`.
   */

  /* This continued fraction expansion of betainc converges rapidly
   * for x < (a - 1) / (a + b - 2). For x >= (a - 1) / (a + b - 2),
   * we can obtain an equivalent computation by using the symmetry
   * relation given here: https://dlmf.nist.gov/8.17.E4
   *   betainc(a, b, x) = 1 - betainc(b, a, 1 - x) */
  Tensor use_symmetry_relation = (x >= (a - 1.0) / (a + b - 2.0));
  const Tensor& _a = at::where(use_symmetry_relation, b, a);
  const Tensor& _b = at::where(use_symmetry_relation, a, b);
  const Tensor& _x = at::where(use_symmetry_relation, 1.0 - x, x);

  const auto&& [cf, cf_grad_a, cf_grad_b] =
      _betainc_modified_lentz_method(_a, _b, _x, use_continued_fraction);

  Tensor normalization = at::exp(
      at::xlogy(_a, _x) + at::special_xlog1py(_b, -_x) - at::log(_a) -
      at::special_betaln(_a, _b));
  Tensor digamma_apb = at::special_digamma(_a + _b);
  Tensor grad_a = normalization *
      (cf_grad_a +
       cf * (at::log(_x) - at::reciprocal(_a) + digamma_apb - at::digamma(_a)));
  Tensor grad_b = normalization *
      (cf_grad_b + cf * (at::log1p(-_x) + digamma_apb - at::digamma(_b)));

  Tensor grad_a_orig = grad_a;
  grad_a = at::where(use_symmetry_relation, -grad_b, grad_a);
  grad_b = at::where(use_symmetry_relation, -grad_a_orig, grad_b);

  return std::make_tuple(std::move(grad_a), std::move(grad_b));
}

static inline std::tuple<Tensor, Tensor> _betainc_der_power_series(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x,
    const Tensor& use_power_series) {
  // Returns the partial derivatives of betainc with respect to a and b.
  /*
   * This function evaluates betainc(a, b, x) by its series representation:
   *   x ** a * 2F1(a, 1 - b; a + 1; x) / (a * B(a, b)) ,
   * where 2F1 is the Gaussian hypergeometric function.
   * We apply this function when the input (a, b, x) satisfies at least one
   * of the following conditions:
   *   C1: (x < a / (a + b)) & (b * x <= 1) & (x <= 0.95)
   *   C2: (x >= a / (a + b)) & (a * (1 - x) <= 1) & (x >= 0.05)
   */
  // a, b and x have same dtype.
  const Tensor eps = AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "__betainc_der_power_series_eps",
      [&]() -> Tensor {
        return at::scalar_tensor(
            std::numeric_limits<
                at::scalar_value_type<scalar_t>::type>::epsilon(),
            x.options());
      });
  auto options = at::TensorOptions().dtype(x.dtype()).device(x.device());
  const Tensor one = at::scalar_tensor(1.0, options);
  const Tensor half = at::scalar_tensor(0.5, options);
  const Tensor _true = at::scalar_tensor(true, x.device());

  // Avoid returning NaN or infinity when the input does not satisfy either C1
  // or C2.
  Tensor safe_a = at::where(use_power_series, a, half);
  Tensor safe_b = at::where(use_power_series, b, half);
  Tensor safe_x = at::where(use_power_series, x, half);

  /* When x >= a / (a + b), we must apply the symmetry relation given here:
   * https://dlmf.nist.gov/8.17.E4
   *   betainc(a, b, x) = 1 - betainc(b, a, 1 - x) */
  Tensor use_symmetry_relation = (safe_x >= safe_a / (safe_a + safe_b));
  Tensor safe_a_orig = safe_a;

  safe_a = at::where(use_symmetry_relation, safe_b, safe_a);
  safe_b = at::where(use_symmetry_relation, safe_a_orig, safe_b);
  safe_x = at::where(use_symmetry_relation, 1.0 - safe_x, safe_x);
  // max_iterations was set by experimentation and tolerance was taken from
  // Cephes.
  int32_t max_iterations = 300; // at::kFloat

  if (x.scalar_type() == at::kDouble) {
    max_iterations = 600;
  }

  Tensor tolerance = eps / safe_a;
  /* Evaluate the series that defines the following expression:
   *   2F1(a, 1 - b; a + 1; x) / a */
  auto __power_series_evaluation = [&](const Tensor& should_stop,
                                       const std::vector<Tensor>& values,
                                       const std::vector<Tensor>& gradients)
      -> std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> {
    const Tensor& n = values.at(0);
    const Tensor& product = values.at(1);
    const Tensor& series_sum = values.at(2);
    const Tensor& product_grad_b = gradients.at(0);
    const Tensor& da = gradients.at(1);
    const Tensor& db = gradients.at(2);

    Tensor x_div_n = safe_x / n;
    Tensor factor = (n - safe_b) * x_div_n;
    Tensor apn = safe_a + n;

    Tensor new_product = product * factor;
    Tensor term = new_product / apn;
    Tensor new_product_grad_b = factor * product_grad_b - product * x_div_n;
    Tensor new_da = da - new_product / at::square(apn);
    Tensor new_db = db + new_product_grad_b / apn;

    Tensor stop = should_stop | (at::abs(term) <= tolerance);
    std::vector<Tensor> new_values = {
        n + 1.0, std::move(new_product), series_sum + term};
    std::vector<Tensor> new_gradients = {
        std::move(new_product_grad_b), std::move(new_da), std::move(new_db)};

    return std::make_tuple(
        std::move(stop), std::move(new_values), std::move(new_gradients));
  };

  Tensor initial_n = one;
  Tensor initial_product = at::ones_like(safe_a);
  Tensor initial_series_sum = one / safe_a;
  std::vector<Tensor> values = {
      std::move(initial_n),
      std::move(initial_product),
      std::move(initial_series_sum)};

  Tensor initial_product_grad_b = at::zeros_like(safe_b);
  Tensor initial_da = -at::reciprocal(at::square(safe_a));
  Tensor initial_db = initial_product_grad_b;
  std::vector<Tensor> gradients = {
      std::move(initial_product_grad_b),
      std::move(initial_da),
      std::move(initial_db)};

  Tensor stop = ~use_power_series;

  for (int32_t i = 0; i < max_iterations; i++) {
    std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> ret =
        __power_series_evaluation(stop, values, gradients);
    stop = std::get<0>(ret);
    values = std::get<1>(ret);
    gradients = std::get<2>(ret);
    if (stop.all().equal(_true)) // TODO: It can be bottleneck..
      break;
  }

  const Tensor& series_sum = values.back();
  const Tensor& series_grad_a = gradients.at(1);
  const Tensor& series_grad_b = gradients.at(2);

  Tensor normalization =
      at::exp(at::xlogy(safe_a, safe_x) - at::special_betaln(safe_a, safe_b));
  Tensor digamma_apb = at::digamma(safe_a + safe_b);
  Tensor grad_a = normalization *
      (series_grad_a +
       series_sum * (digamma_apb - at::digamma(safe_a) + at::log(safe_x)));
  Tensor grad_b = normalization *
      (series_grad_b + series_sum * (digamma_apb - at::digamma(safe_b)));

  Tensor grad_a_orig = grad_a;
  grad_a = at::where(use_symmetry_relation, -grad_b, grad_a);
  grad_b = at::where(use_symmetry_relation, -grad_a_orig, grad_b);

  return std::make_tuple(std::move(grad_a), std::move(grad_b));
}

static inline std::tuple<Tensor, Tensor, Tensor> _betainc_partials(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x) {
  // Reference: https://github.com/tensorflow/probability/blob/
  // b14ae1d79de4a52d834a3ba2dc88f7e5d849e6c7/tensorflow_probability/python/math/special.py#L432-L491
  at::ScalarType dtype_origin = at::promoteTypes(
      at::promoteTypes(a.scalar_type(), b.scalar_type()), x.scalar_type());
  /* We promote bfloat16 and float16 to float32 to make this function consistent
   * with betainc */
  bool should_promote_dtype = ((dtype_origin == at::ScalarType::BFloat16) |
                               (dtype_origin == at::ScalarType::Half))
      ? true
      : false;
  at::ScalarType dtype =
      should_promote_dtype ? at::ScalarType::Float : dtype_origin;

  const Tensor& _a = a.to(dtype);
  const Tensor& _b = b.to(dtype);
  const Tensor& _x = x.to(dtype);

  /* The partial derivative of betainc with respect to x can be obtained
   * directly by using the expression given here:
   * http://functions.wolfram.com/06.21.20.0001.01 */
  Tensor grad_x = at::exp(
      at::xlogy(_a - 1.0, _x) + at::special_xlog1py(_b - 1.0, -_x) -
      at::special_betaln(_a, _b));

  /* The partial derivatives of betainc with respect to a and b are computed
   * by using forward mode. */
  Tensor use_power_series =
      (((_x < _a / (_a + _b)) & (_b * _x <= 1.0) & (_x <= 0.95)) |
       ((_x >= _a / (_a + _b)) & (_a * (1.0 - _x) <= 1.0) & (_x >= 0.05)));
  const auto&& [ps_grad_a, ps_grad_b] =
      _betainc_der_power_series(_a, _b, _x, use_power_series);

  const auto&& [cf_grad_a, cf_grad_b] =
      _betainc_der_continued_fraction(_a, _b, _x, ~use_power_series);

  Tensor grad_a = at::where(use_power_series, ps_grad_a, cf_grad_a);
  Tensor grad_b = at::where(use_power_series, ps_grad_b, cf_grad_b);

  /* According to the code accompanying [1], grad_a = grad_b = 0 when x is
   * equal to 0 or 1.
   * [1] R. Boik, J. Robinson-Cox,
   *    Derivatives of the Incomplete Beta Function
   *    https://www.jstatsoft.org/article/view/v003i01/beta.der.pdf */
  Tensor grads_a_and_b_should_be_zero = (_x == 0.0) | (_x == 1.0);

  grad_a = at::where(grads_a_and_b_should_be_zero, 0.0, grad_a);
  grad_b = at::where(grads_a_and_b_should_be_zero, 0.0, grad_b);

  // Determine if the inputs are out of range (should return NaN output).
  Tensor result_is_nan = (a <= 0.0) | (b <= 0.0) | (x < 0.0) | (x > 1.0);
  const Tensor nan = AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "__betainc_partials_nan",
      [&]() -> Tensor {
        return at::scalar_tensor(
            std::numeric_limits<
                at::scalar_value_type<scalar_t>::type>::quiet_NaN(),
            x.options());
      });

  grad_a = at::where(result_is_nan, nan, grad_a);
  grad_b = at::where(result_is_nan, nan, grad_b);
  grad_x = at::where(result_is_nan, nan, grad_x);

  /* If we promoted the dtype, then we have to convert the gradients back to the
   * original dtype. */
  if (should_promote_dtype) {
    grad_a = grad_a.to(dtype_origin);
    grad_b = grad_b.to(dtype_origin);
    grad_x = grad_x.to(dtype_origin);
  }
  return std::make_tuple(
      std::move(grad_a), std::move(grad_b), std::move(grad_x));
}

TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betainc_partials(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x) {
  return _betainc_partials(a, b, x);
}

static inline std::tuple<Tensor, Tensor, Tensor> _betaincinv_partials(
    const Tensor& a,
    const Tensor& b,
    const Tensor& y) {
  at::ScalarType dtype_orig = at::promoteTypes(
      at::promoteTypes(a.scalar_type(), b.scalar_type()), y.scalar_type());
  bool should_promote_dtype = ((dtype_orig == at::ScalarType::BFloat16) |
                               (dtype_orig == at::ScalarType::Half))
      ? true
      : false;
  at::ScalarType dtype =
      should_promote_dtype ? at::ScalarType::Float : dtype_orig;
  Tensor _y = y.to(dtype_orig);
  Tensor _a = a.to(dtype_orig);
  Tensor _b = b.to(dtype_orig);

  if (should_promote_dtype) {
    _y = _y.to(dtype);
    _a = _a.to(dtype);
    _b = _b.to(dtype);
  }

  Tensor _x = at::special_betaincinv(_a, _b, _y);
  auto [g_a, g_b, g_x] = _betainc_partials(_a, _b, _x);
  g_a = -g_a / g_x;
  g_b = -g_b / g_x;
  Tensor g_y = at::reciprocal(g_x);

  if (should_promote_dtype) {
    g_a = g_a.to(dtype_orig);
    g_b = g_b.to(dtype_orig);
    g_y = g_y.to(dtype_orig);
  }

  return std::make_tuple(std::move(g_a), std::move(g_b), std::move(g_y));
}

TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betaincinv_partials(
    const Tensor& a,
    const Tensor& b,
    const Tensor& y) {
  return _betaincinv_partials(a, b, y);
}


std::tuple<Tensor, Tensor, Tensor> _special_betainc_partials_meta(
    const Tensor& a,
    const Tensor& b,
    const Tensor& x) {
  at::ScalarType dtype_orig = at::promoteTypes(
      at::promoteTypes(a.scalar_type(), b.scalar_type()), x.scalar_type());
  auto vec = at::broadcast_tensors({a, b, x});

  return std::make_tuple(at::empty_like(vec.at(0), at::MemoryFormat::Contiguous).to(dtype_orig),
      at::empty_like(vec.at(1), at::MemoryFormat::Contiguous).to(dtype_orig),
      at::empty_like(vec.at(2), at::MemoryFormat::Contiguous).to(dtype_orig));
}


std::tuple<Tensor, Tensor, Tensor> _special_betaincinv_partials_meta(
    const Tensor& a,
    const Tensor& b,
    const Tensor& y) {
  at::ScalarType dtype_orig = at::promoteTypes(
      at::promoteTypes(a.scalar_type(), b.scalar_type()), y.scalar_type());

  auto vec = at::broadcast_tensors({a, b, y});

  return std::make_tuple(at::empty_like(vec.at(0), at::MemoryFormat::Contiguous).to(dtype_orig),
      at::empty_like(vec.at(1), at::MemoryFormat::Contiguous).to(dtype_orig),
      at::empty_like(vec.at(2), at::MemoryFormat::Contiguous).to(dtype_orig));

}


} // namespace at::native
