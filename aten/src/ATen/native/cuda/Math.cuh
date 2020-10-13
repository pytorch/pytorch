#pragma once

#include <ATen/AccumulateType.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

/*
 * For licensing information, please refer to the the cpu implementation located in "ATen/native/Math.h".
 */
template <typename scalar_t>
static inline __host__ __device__ scalar_t zeta(scalar_t _x, scalar_t _q) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  static const accscalar_t MACHEP = 1.11022302462515654042E-16;
  const accscalar_t A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };
  accscalar_t x = static_cast<accscalar_t>(_x);
  accscalar_t q = static_cast<accscalar_t>(_q);

  int i = 0;
  accscalar_t a, b, k, s, t, w;
  if( x == 1.0 ) {
    return static_cast<scalar_t>(INFINITY);
  }

  if( x < 1.0 ){
    std::numeric_limits<scalar_t>::quiet_NaN();
  }
  bool q_is_integer = q == ::floor(q);

  if(q <= 0.0) {
    if(q_is_integer) {
      return static_cast<scalar_t>(INFINITY);
    }
    else {
      std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }

  s = ::pow(q, -x);
  a = q;
  i = 0;
  b = 0.0;
  while((i < 9) || (a <= 9.0)){
    i += 1;
    a += 1.0;
    b = ::pow( a, -x );
    s += b;
    if((-MACHEP < (b / s)) && ((b / s) < MACHEP)) {
      return static_cast<scalar_t>(s);
    }
  };
  w = a;
  s += b * w / (x - 1.0);
  s -= 0.5 * b;
  a = 1.0;
  k = 0.0;
  for(int i=0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = t / s;
    if(t < 0){
      t = -t;
    }
    if((-MACHEP <t) && (t < MACHEP)){
      return static_cast<scalar_t>(s);
    }
    k += 1.0;
    a *= x + k;
    b /= w;
    k += 1.0;
  }
  return static_cast<scalar_t>(s);
}

/*
 * For licensing information, please refer to the the cpu implementation located in "ATen/native/Math.h".
 */
template <typename scalar_t>
static inline __host__ __device__ scalar_t calc_digamma(scalar_t in) {
  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  static const double PI_f64 = 3.14159265358979323846;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  const accscalar_t A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t x = static_cast<accscalar_t>(in);
  if (x == 0) {
    return static_cast<scalar_t>(INFINITY);
  }

  bool x_is_integer = x == ::floor(x);
  accscalar_t result = 0;
  if (x < 0) {
    if (x_is_integer) {
      return static_cast<scalar_t>(INFINITY);
    }
    // Rounding errors in tan's input can really affect the output
    // for extreme values, so we always perform this computation in double.
    result = static_cast<accscalar_t>(- PI_f64 / ::tan(PI_f64 * static_cast<double>(x)));
    x = 1 - x;
  }

  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return static_cast<scalar_t>(result + PSI_10);
  }

  accscalar_t y = 0;
  if (x < 1.0e17) {
    accscalar_t z = 1 / (x * x);

    accscalar_t polevl_result = 0;
    for (int i = 0; i <= 6; i++) {
      polevl_result = polevl_result * z + A[i];
    }
    y = z * polevl_result;
  }

  return static_cast<scalar_t>(::log(x) - (static_cast<accscalar_t>(0.5) / x) - y + result);
}

template <typename scalar_t>
static inline __host__ __device__ scalar_t calc_trigamma(scalar_t in) {
  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  const accscalar_t PI = 3.14159265358979323846;
  accscalar_t x = static_cast<accscalar_t>(in);
  accscalar_t sign = +1;
  accscalar_t result = 0;
  if (x < 0.5f) {
    sign = -1;
    accscalar_t sin_pi_x = ::sin(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const accscalar_t one = static_cast<scalar_t>(1);
  const accscalar_t ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (one/6 - ixx * (one/30 - ixx * (one/42)))) / x;
  return static_cast<scalar_t>(sign * result);
}

template <typename scalar_t>
static inline __host__ __device__ scalar_t calc_polygamma(int n, scalar_t x) {
  // already blocked if n <= 1
  return ((n % 2) ? 1.0 : -1.0) * ::exp(::lgamma(static_cast<scalar_t>(n) + 1.0)) * zeta(static_cast<scalar_t>(n + 1), x);
}

template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t _igamma_frac(scalar_t a, scalar_t x) {

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t ans, ax, c, r;
  static accscalar_t MAXLOG = std::is_same<accscalar_t, double>::value ?
    7.09782712893383996843E2 : 88.72283905206835;
  static accscalar_t MACHEP = std::is_same<accscalar_t, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;

  /* Compute  x**a * exp(-x) / gamma(a)  */
  ax = a * ::log(x) - x - ::lgamma(a);
  if(ax < -MAXLOG) {
    return 0.0; // underflow
  }
  ax = ::exp(ax);

  /* power series */
  r = a;
  c = 1.0;
  ans = 1.0;

  do {
    r += 1.0;
    c *= x / r;
    ans += c;
  }
  while (c > MACHEP * ans);

  return ans * ax / a;
}

template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t _igammac_frac(scalar_t a, scalar_t x) {

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t ans, ax, c, yc, r, t, y, z;
  accscalar_t pk, pkm1, pkm2, qk, qkm1, qkm2;
  static accscalar_t MAXLOG = std::is_same<accscalar_t,double>::value ?
    7.09782712893383996843E2 : 88.72283905206835;
  static accscalar_t MACHEP = std::is_same<accscalar_t,double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  static accscalar_t BIG = std::is_same<accscalar_t,double>::value ?
    4.503599627370496e15 : 16777216.;
  static accscalar_t BIGINV = std::is_same<accscalar_t,double>::value ?
    2.22044604925031308085e-16 : 5.9604644775390625E-8;

  ax = a * ::log(x) - x - ::lgamma(a);
  if (ax < -MAXLOG) {
    return 0.0; // underflow
  }
  ax = ::exp(ax);

  /* continued fraction */
  y = 1.0 - a;
  z = x + y + 1.0;
  c = 0.0;
  pkm2 = 1.0;
  qkm2 = x;
  pkm1 = x + 1.0;
  qkm1 = z * x;
  ans = pkm1/qkm1;

  do {
    c += 1.0;
    y += 1.0;
    z += 2.0;
    yc = y * c;
    pk = pkm1 * z  -  pkm2 * yc;
    qk = qkm1 * z  -  qkm2 * yc;
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
  }
  while (t > MACHEP);

  return ans * ax;
}

template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t calc_igamma(scalar_t a, scalar_t x) {
  // boundary values following SciPy
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 1.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 0.0; // zero integration limit
  }
  else if (::isinf(a)) {
    if (::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 0.0;
  }
  else if (::isinf(x)) {
    return 1.0;
  }

  if ((x > 1.0) && (x > a)) {
    return 1.0 - _igammac_frac(a, x);
  }
  return _igamma_frac(a, x);
}

template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t calc_igammac(scalar_t a, scalar_t x) {
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 0.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 1.0;
  }
  else if (::isinf(a)) {
    if (::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 1.0;
  }
  else if (::isinf(x)) {
    return 0.0;
  }

  if ((x < 1.0) || (x < a)) {
    return 1.0 - _igamma_frac(a, x);
  }
  return _igammac_frac(a, x);
}

template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t calc_gcd(scalar_t a_in, scalar_t b_in) {
  scalar_t a = ::abs(a_in);
  scalar_t b = ::abs(b_in);
  while (a != 0) {
    scalar_t c = a;
    a = b % a;
    b = c;
  }
  return b;
}

/*
 * For licensing information and documentation, please refer to the the cpu implementation located in "ATen/native/Math.h".
 */
template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t chbevl(scalar_t _x, const scalar_t array[], size_t len) {
  using accscalar_t = at::acc_type<scalar_t, true>;

  accscalar_t x = static_cast<accscalar_t>(_x);
  accscalar_t b0, b1, b2;

  b0 = static_cast<accscalar_t>(array[0]);
  b1 = 0;

  for (size_t i = 1; i < len; ++i)  {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + static_cast<accscalar_t>(array[i]);
  }

  return static_cast<scalar_t>(0.5 * (b0 - b2));
}

/*
 * For licensing information and documentation, please refer to the the cpu implementation located in "ATen/native/Math.h".
 */
template <typename scalar_t>
static inline C10_HOST_DEVICE scalar_t calc_i0(scalar_t _x) {
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Upcast input for numerical accuracy purposes
  // Needed for accurate results if input is bfloat16 or float16
  accscalar_t x = ::abs(static_cast<accscalar_t>(_x));

  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  const accscalar_t A[] = {
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
  };

  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  const accscalar_t B[] = {
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
  };

  if (x <= 8.0) {
    accscalar_t y = static_cast<accscalar_t>((x / 2.0) - 2.0);
    return static_cast<scalar_t>(::exp(x) * chbevl(y, A, 30));
  }

  return static_cast<scalar_t>(::exp(x) * chbevl(static_cast<accscalar_t>(32.0 / x - 2.0), B, 25) / ::sqrt(x));
}

}
}
