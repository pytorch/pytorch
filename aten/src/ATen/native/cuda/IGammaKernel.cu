#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/IGammaCoefficients.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

// TODO: review jiterating igamma and igammac if/when a persistent (across processes)
//   cache is implemented, because they take a VERY long time to compile
// TODO: it's also odd these ops use gpu_kernel_with_scalars

namespace {

/*
 * This implementation of the regularized incomplete gamma functions and
 * their helper functions are derived from the implementation of SciPy's
 * gammainc, Cephes's igam and igamc, and Boost's Lanczos approximations.
 * See NOTICE for the licenses.
 */
// regularized lower & upper incomplete gamma
template <typename scalar_t>
__host__ __device__ scalar_t ratevl(scalar_t x, const scalar_t num[], int64_t M,
    const scalar_t denom[], int64_t N) {
  // evaluating rational function, i.e., the ratio of two polynomials
  // the coefficients for numerator are given by `num` while coeffs for
  // denumerator are given by `denom`

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  int64_t i, dir;
  accscalar_t y, num_ans, denom_ans;
  accscalar_t absx = ::fabs(x);
  const accscalar_t *p;

  if (absx > 1) {
    /* Evaluate as a polynomial in 1/x. */
    dir = -1;
    p = num + M;
    y = 1 / x;
  }
  else {
    dir = 1;
    p = num;
    y = x;
  }

  /* Evaluate the numerator */
  num_ans = *p;
  p += dir;
  for (i = 1; i <= M; i++) {
    num_ans = num_ans * y + *p;
    p += dir;
  }
  /* Evaluate the denominator */
  if (absx > 1) {
    p = denom + N;
  }
  else {
    p = denom;
  }

  denom_ans = *p;
  p += dir;
  for (i = 1; i <= N; i++) {
    denom_ans = denom_ans * y + *p;
    p += dir;
  }
  if (absx > 1) {
    i = N - M;
    return ::pow(x, static_cast<accscalar_t>(i)) * num_ans / denom_ans;
  }
  else {
    return num_ans / denom_ans;
  }
}

template <typename scalar_t>
__host__ __device__ scalar_t lanczos_sum_expg_scaled(scalar_t x) {
  // lanczos approximation
  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

  constexpr accscalar_t lanczos_sum_expg_scaled_num[13] = {
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
  };
  constexpr accscalar_t lanczos_sum_expg_scaled_denom[13] = {
    1.,
    66.,
    1925.,
    32670.,
    357423.,
    2637558.,
    13339535.,
    45995730.,
    105258076.,
    150917976.,
    120543840.,
    39916800.,
    0
  };
  return ratevl(static_cast<accscalar_t>(x), lanczos_sum_expg_scaled_num,
      sizeof(lanczos_sum_expg_scaled_num) / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
      lanczos_sum_expg_scaled_denom,
      sizeof(lanczos_sum_expg_scaled_denom) / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}

template <typename scalar_t>
__host__ __device__ scalar_t _igam_helper_fac(scalar_t a, scalar_t x) {
  // compute x^a * exp(-a) / gamma(a)
  // corrected from (15) and (16) in [igam2] by replacing exp(x - a) with
  // exp(a - x).

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t ax, fac, res, num, numfac;
  constexpr accscalar_t MAXLOG = std::is_same_v<accscalar_t,double> ?
    7.09782712893383996843E2 : 88.72283905206835;
  constexpr accscalar_t EXP1 = 2.718281828459045;
  constexpr accscalar_t lanczos_g = 6.024680040776729583740234375;

  if (::fabs(a - x) > 0.4 * ::fabs(a)) {
    ax = a * ::log(x) - x - ::lgamma(a);
    if (ax < -MAXLOG) {
      return 0.0;
    }
    return ::exp(ax);
  }

  fac = a + lanczos_g - 0.5;
  res = ::sqrt(fac / EXP1) / lanczos_sum_expg_scaled(a);

  if ((a < 200) && (x < 200)) {
    res *= ::exp(a - x) * ::pow(x / fac, a);
  }
  else {
    num = x - a - lanczos_g + 0.5;
    numfac = num / fac;
    res *= ::exp(a * (::log1p(numfac) - numfac) + x * (0.5 - lanczos_g) / fac);
  }
  return res;
}

template <typename scalar_t>
__host__ __device__ scalar_t _igam_helper_series(scalar_t a, scalar_t x) {
  // Compute igam using DLMF 8.11.4. [igam1]

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  constexpr accscalar_t MACHEP = std::is_same_v<accscalar_t, double> ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  constexpr int MAXITER = 2000;

  int i;
  accscalar_t ans, ax, c, r;

  ax = _igam_helper_fac(a, x);
  if (ax == 0.0) {
    return 0.0;
  }

  /* power series */
  r = a;
  c = 1.0;
  ans = 1.0;

  for (i = 0; i < MAXITER; i++) {
    r += 1.0;
    c *= x / r;
    ans += c;
    if (c <= MACHEP * ans) {
      break;
    }
  }
  return (ans * ax / a);
}

template <typename scalar_t>
__host__ __device__ scalar_t _igamc_helper_series(scalar_t a, scalar_t x) {
  // Compute igamc using DLMF 8.7.3 [igam1]. This is related to the series in
  // _igam_helper_series but extra care is taken to avoid cancellation.

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  int n;
  accscalar_t fac = 1;
  accscalar_t sum = 0;
  accscalar_t term, logx;
  constexpr int MAXITER = 2000;
  constexpr accscalar_t MACHEP = std::is_same_v<accscalar_t, double> ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;

  for (n = 1; n < MAXITER; n++) {
    fac *= -x / n;
    term = fac / (a + n);
    sum += term;
    if (::fabs(term) <= MACHEP * ::fabs(sum)) {
        break;
    }
  }

  logx = ::log(x);
  term = -::expm1(a * logx - ::lgamma(1+a));
  return term - ::exp(a * logx - ::lgamma(a)) * sum;
}

template <typename scalar_t>
__host__ __device__ scalar_t _igam_helper_asymptotic_series(scalar_t a, scalar_t x, bool igam) {
  // Compute igam/igamc using DLMF 8.12.3/8.12.4 [igam1]

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  constexpr accscalar_t d[25][25] =
    ATEN_IGAMMA_ASYMPTOTIC_COEFFICIENTS;

  int k, n, sgn;
  int maxpow = 0;
  constexpr accscalar_t MACHEP = std::is_same_v<accscalar_t, double> ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  accscalar_t lambda = x / a;
  accscalar_t sigma = (x - a) / a;
  accscalar_t eta, res, ck, ckterm, term, absterm;
  accscalar_t absoldterm = INFINITY;
  accscalar_t etapow[25] = {1};
  accscalar_t sum = 0;
  accscalar_t afac = 1;

  if (igam) {
    sgn = -1;
  }
  else {
    sgn = 1;
  }

  if (lambda > 1) {
    eta = ::sqrt(-2 * (::log1p(sigma) - sigma));
  }
  else if (lambda < 1) {
    eta = -::sqrt(-2 * (::log1p(sigma) - sigma));
  }
  else {
    eta = 0;
  }
  res = 0.5 * ::erfc(sgn * eta * ::sqrt(a / 2));

  for (k = 0; k < 25; k++) {
    ck = d[k][0];
    for (n = 1; n < 25; n++) {
      if (n > maxpow) {
        etapow[n] = eta * etapow[n-1];
        maxpow += 1;
      }
      ckterm = d[k][n]*etapow[n];
      ck += ckterm;
      if (std::fabs(ckterm) < MACHEP * std::fabs(ck)) {
        break;
      }
    }
    term = ck * afac;
    absterm = std::fabs(term);
    if (absterm > absoldterm) {
      break;
    }
    sum += term;
    if (absterm < MACHEP * std::fabs(sum)) {
      break;
    }
    absoldterm = absterm;
    afac /= a;
  }
  res += sgn * ::exp(-0.5 * a * eta * eta) * sum / ::sqrt(2 * 3.1415926535 * a);

  return res;
}

template <typename scalar_t>
__host__ __device__ scalar_t _igamc_helper_continued_fraction(scalar_t a, scalar_t x) {
  // Compute igamc using DLMF 8.9.2. [igam1]

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  int i;
  accscalar_t ans, ax, c, yc, r, t, y, z;
  accscalar_t pk, pkm1, pkm2, qk, qkm1, qkm2;
  constexpr int MAXITER = 2000;
  constexpr accscalar_t MACHEP = std::is_same_v<accscalar_t, double> ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  constexpr accscalar_t BIG = std::is_same_v<accscalar_t,double> ?
    4.503599627370496e15 : 16777216.;
  constexpr accscalar_t BIGINV = std::is_same_v<accscalar_t,double> ?
    2.22044604925031308085e-16 : 5.9604644775390625E-8;

  ax = _igam_helper_fac(a, x);
  if (ax == 0.0) {
    return 0.0;
  }

  /* continued fraction */
  y = 1.0 - a;
  z = x + y + 1.0;
  c = 0.0;
  pkm2 = 1.0;
  qkm2 = x;
  pkm1 = x + 1.0;
  qkm1 = z * x;
  ans = pkm1 / qkm1;

  for (i = 0; i < MAXITER; i++) {
    c += 1.0;
    y += 1.0;
    z += 2.0;
    yc = y * c;
    pk = pkm1 * z - pkm2 * yc;
    qk = qkm1 * z - qkm2 * yc;
    if (qk != 0) {
      r = pk / qk;
      t = ::fabs((ans - r) / r);
      ans = r;
    }
    else {
      t = 1.0;
    }
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    if (::fabs(pk) > BIG) {
      pkm2 *= BIGINV;
      pkm1 *= BIGINV;
      qkm2 *= BIGINV;
      qkm1 *= BIGINV;
    }
    if (t <= MACHEP) {
      break;
    }
  }
  return ans * ax;
}

template <typename scalar_t>
__noinline__ __host__ __device__ scalar_t calc_igammac(scalar_t a, scalar_t x) {
  /* the calculation of the regularized upper incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.4 [igam1])
   * - if x > 1.1 and x < a, using the subtraction from the regularized lower
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (5)
   */

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t absxma_a;

  constexpr accscalar_t SMALL = 20.0;
  constexpr accscalar_t LARGE = 200.0;
  constexpr accscalar_t SMALLRATIO = 0.3;
  constexpr accscalar_t LARGERATIO = 4.5;

  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<accscalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 0.0;
    }
    else {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 1.0;
  }
  else if (::isinf(static_cast<accscalar_t>(a))) {
    if (::isinf(static_cast<accscalar_t>(x))) {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
    return 1.0;
  }
  else if (::isinf(static_cast<accscalar_t>(x))) {
    return 0.0;
  }

  absxma_a = ::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / ::sqrt(a))) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }

  if (x > 1.1) {
    if (x < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_continued_fraction(a, x);
    }
  }
  else if (x <= 0.5) {
    if (-0.4 / ::log(x) < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
  else {
    if (x * 1.1 < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
}

// NOTE: this __noinline__ is important -- otherwise, observed compile times significantly
// increase.  The same kernel seems to get recompiled multiple times via gpu_kernel_with_scalars,
// multiple dtypes, etc.
template <typename scalar_t>
__noinline__ __host__ __device__ scalar_t calc_igamma(scalar_t a, scalar_t x) {
  /* the calculation of the regularized lower incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.3 [igam1])
   * - if x > 1 and x > a, using the subtraction from the regularized upper
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (4)
   */

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t absxma_a;
  constexpr accscalar_t SMALL = 20.0;
  constexpr accscalar_t LARGE = 200.0;
  constexpr accscalar_t SMALLRATIO = 0.3;
  constexpr accscalar_t LARGERATIO = 4.5;

  // boundary values following SciPy
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<accscalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 1.0;
    }
    else {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 0.0; // zero integration limit
  }
  else if (::isinf(static_cast<accscalar_t>(a))) {
    if (::isinf(static_cast<accscalar_t>(x))) {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
    return 0.0;
  }
  else if (::isinf(static_cast<accscalar_t>(x))) {
    return 1.0;
  }

  /* Asymptotic regime where a ~ x. */
  absxma_a = ::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / ::sqrt(a))) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }

  if ((x > 1.0) && (x > a)) {
    return 1.0 - calc_igammac(a, x);
  }

  return _igam_helper_series(a, x);
}

template<typename scalar_t>
struct CalcIgamma{
  CalcIgamma(bool calc_igammac): calc_igammac_(calc_igammac){}
  bool calc_igammac_;
  __device__ scalar_t operator() (scalar_t a, scalar_t b) const {
    if (calc_igammac_) {
      return calc_igammac(a,b);
    } else {
      return calc_igamma(a,b);
    }
  }
};

}

// end of regularized lower & upper incomplete gamma

namespace at::native {

void igamma_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_cuda", [&]() {
    gpu_kernel(iter, CalcIgamma<scalar_t>(false));
  });
}

void igammac_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_cuda", [&]() {
    gpu_kernel(iter, CalcIgamma<scalar_t>(true));
  });
}

REGISTER_DISPATCH(igamma_stub, &igamma_kernel_cuda)
REGISTER_DISPATCH(igammac_stub, &igammac_kernel_cuda)

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

} // namespace at::native
