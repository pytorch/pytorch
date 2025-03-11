// Implementation of specal math functions for Metal
#pragma once
#include <c10/metal/utils.h>
#include <metal_stdlib>

namespace c10 {
namespace metal {

// Translated to metal from https://www.johndcook.com/cpp_erf.html

template <typename T>
inline T erf(T x) {
  T a1 = 0.254829592;
  T a2 = -0.284496736;
  T a3 = 1.421413741;
  T a4 = -1.453152027;
  T a5 = 1.061405429;
  T p = 0.3275911;

  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = ::metal::fabs(x);

  // A&S formula 7.1.26
  T t = 1.0 / (1.0 + p * x);
  T y = 1.0 -
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
          ::metal::exp(-x * x);

  return sign * y;
}

template <typename T>
inline float erfinv(T y) {
  /* coefficients in rational expansion */
  constexpr float a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
  constexpr float b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
  constexpr float c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
  constexpr float d[2] = {3.543889200, 1.637067800};

  float x, z, num, dem; /*working variables */

  float y_abs = ::metal::abs(static_cast<float>(y));
  if (y_abs >= 1.0f) {
    return y_abs > 1.0f ? NAN
                        : ::metal::copysign(INFINITY, static_cast<float>(y));
  }
  if (y_abs <= 0.7f) {
    z = y * y;
    num = ((a[3] * z + a[2]) * z + a[1]) * z + a[0];
    dem = (((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0f;
    x = y * num / dem;
  } else {
    z = ::metal::sqrt(-1.0f * ::metal::log((1.0 - y_abs) / 2.0));
    num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
    dem = (d[1] * z + d[0]) * z + 1.0f;
    x = ::metal::copysign(num, static_cast<float>(y)) / dem;
  }

  return x;
}

/*
 * For licensing information and documentation, please refer to the cpu
 * implementation located in "ATen/native/Math.h".
 */

template <typename T>
inline T chbevl(T x, const float array[], const int len) {
  T b0, b1, b2;

  b0 = array[0];
  b1 = 0;

  for (int i = 1; i < len; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + array[i];
  }

  return T{0.5} * (b0 - b2);
}

// Copied from
// https://github.com/pytorch/pytorch/blob/58b661cda2c002a8e1ac3bee494bfe1f7420437c/aten/src/ATen/native/cuda/Math.cuh#L502

template <typename T>
inline T i0(T _x) {
  auto x = ::metal::fabs(_x);

  if (x <= 8.0) {
    /* Chebyshev coefficients for exp(-x) I0(x)
     *   in the interval [0,8].
     *
     * lim(x->0){ exp(-x) I0(x) } = 1.
     */
    constexpr float A[] = {
        -4.41534164647933937950E-18, 3.33079451882223809783E-17,
        -2.43127984654795469359E-16, 1.71539128555513303061E-15,
        -1.16853328779934516808E-14, 7.67618549860493561688E-14,
        -4.85644678311192946090E-13, 2.95505266312963983461E-12,
        -1.72682629144155570723E-11, 9.67580903537323691224E-11,
        -5.18979560163526290666E-10, 2.65982372468238665035E-9,
        -1.30002500998624804212E-8,  6.04699502254191894932E-8,
        -2.67079385394061173391E-7,  1.11738753912010371815E-6,
        -4.41673835845875056359E-6,  1.64484480707288970893E-5,
        -5.75419501008210370398E-5,  1.88502885095841655729E-4,
        -5.76375574538582365885E-4,  1.63947561694133579842E-3,
        -4.32430999505057594430E-3,  1.05464603945949983183E-2,
        -2.37374148058994688156E-2,  4.93052842396707084878E-2,
        -9.49010970480476444210E-2,  1.71620901522208775349E-1,
        -3.04682672343198398683E-1,  6.76795274409476084995E-1};

    auto y = (x / 2.0) - 2.0;
    return static_cast<T>(::metal::exp(x) * chbevl(y, A, 30));
  }

  // Handles x > 8 case
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  constexpr float B[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return static_cast<T>(
      (::metal::exp(x) * chbevl(32.0 / x - 2.0, B, 25)) / ::metal::sqrt(x));
}

// Copied from
// https://github.com/pytorch/pytorch/blob/58b661cda2c002a8e1ac3bee494bfe1f7420437c/aten/src/ATen/native/cuda/Math.cuh#L576

template <typename T>
inline T i1(T _x) {
  const auto x = ::metal::fabs(_x);

  if (x <= 8.0) {
    // Chebyshev coefficients for exp(-x) i1(x) in the internal [0, 8]
    //   lim(x->0){ exp(-x) i1(x) / x } = 1/2
    constexpr float coefficients[] = {
        2.77791411276104639959E-18, -2.11142121435816608115E-17,
        1.55363195773620046921E-16, -1.10559694773538630805E-15,
        7.60068429473540693410E-15, -5.04218550472791168711E-14,
        3.22379336594557470981E-13, -1.98397439776494371520E-12,
        1.17361862988909016308E-11, -6.66348972350202774223E-11,
        3.62559028155211703701E-10, -1.88724975172282928790E-9,
        9.38153738649577178388E-9,  -4.44505912879632808065E-8,
        2.00329475355213526229E-7,  -8.56872026469545474066E-7,
        3.47025130813767847674E-6,  -1.32731636560394358279E-5,
        4.78156510755005422638E-5,  -1.61760815825896745588E-4,
        5.12285956168575772895E-4,  -1.51357245063125314899E-3,
        4.15642294431288815669E-3,  -1.05640848946261981558E-2,
        2.47264490306265168283E-2,  -5.29459812080949914269E-2,
        1.02643658689847095384E-1,  -1.76416518357834055153E-1,
        2.52587186443633654823E-1};
    const auto y = x / 2.0 - 2.0;
    const auto out = ::metal::exp(x) * x * chbevl(y, coefficients, 29);
    return static_cast<T>(_x < T(0.) ? -out : out);
  }

  // Chebyshev coefficients for exp(-x) sqrt(x) i1(x)
  //   in the inverted interval [8, infinity]
  //   lim(x->inf){ exp(-x) sqrt(x) i1(x) } = 1/sqrt(2pi)
  constexpr float coefficients[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};
  const auto out = (::metal::exp(x) * chbevl(32. / x - 2., coefficients, 25)) /
      ::metal::sqrt(x);
  return static_cast<T>(_x < T(0.) ? -out : out);
}

// gamma, lgamma
template <typename T>
inline float log_gamma(const T);

template <typename T>
inline float gamma(const T x) {
  if (x < 0.001) {
    constexpr float EULER_MASCHERONI = 0.577215664901532860606512090;
    // For small x, 1/gamma(x) has power series x + gamma x^2  - ...
    // So in this range, 1/gamma(x) = x + gamma x^2 with error on the order of
    // x^3. The relative error over this interval is less than 6e-7.

    return 1.0 / (x * (1.0 + EULER_MASCHERONI * x));
  }
  if (x >= 12.0) {
    return ::metal::exp(log_gamma(x));
  }
  // The algorithm directly approximates gamma over (1,2) and uses
  // reduction identities to reduce other arguments to this interval.
  // numerator coefficients for gamma approximation over the interval (1,2)
  constexpr float GAMMA_NUMERATOR_COEF[8] = {
      -1.71618513886549492533811E+0,
      2.47656508055759199108314E+1,
      -3.79804256470945635097577E+2,
      6.29331155312818442661052E+2,
      8.66966202790413211295064E+2,
      -3.14512729688483675254357E+4,
      -3.61444134186911729807069E+4,
      6.64561438202405440627855E+4};

  // denominator coefficients for gamma approximation over the interval (1,2)
  constexpr float GAMMA_DENOMINATOR_COEF[8] = {
      -3.08402300119738975254353E+1,
      3.15350626979604161529144E+2,
      -1.01515636749021914166146E+3,
      -3.10777167157231109440444E+3,
      2.25381184209801510330112E+4,
      4.75584627752788110767815E+3,
      -1.34659959864969306392456E+5,
      -1.15132259675553483497211E+5};

  // Add or subtract integers as necessary to bring y into (1,2)
  float y = 1.0 + ::metal::fract(x);

  float num = 0.0;
  float den = 1.0;

  float z = y - 1;
  for (int i = 0; i < 8; i++) {
    num = (num + GAMMA_NUMERATOR_COEF[i]) * z;
    den = den * z + GAMMA_DENOMINATOR_COEF[i];
  }
  float result = num / den + 1.0;

  // Apply correction if argument was not initially in (1,2)
  if (x < 1.0) {
    // identity gamma(z) = gamma(z+1)/z
    result /= (y - 1.0);
  } else {
    // identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
    auto n = static_cast<int>(::metal::floor(x));
    for (int i = 1; i < n; i++) {
      result *= y++;
    }
  }

  return result;
}

template <typename T>
inline float log_gamma(const T x) {
  constexpr float LOG_PI = 1.14472988584940017414342735135305;
  constexpr float HALF_LOG_TWO_PI = 0.91893853320467274178032973640562;
  constexpr float LGAMMA_EXPANSION_COEF[8] = {
      1.0 / 12.0,
      -1.0 / 360.0,
      1.0 / 1260.0,
      -1.0 / 1680.0,
      1.0 / 1188.0,
      -691.0 / 360360.0,
      1.0 / 156.0,
      -3617.0 / 122400.0};

  float rc;

  const auto abs_x = ::metal::abs(static_cast<float>(x));
  if (abs_x == 0) {
    return INFINITY;
  }
  if (abs_x < 12.0) {
    rc = ::metal::log(::metal::abs(gamma(abs_x)));
  } else {
    // Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    float z = 1.0 / (abs_x * abs_x);
    float sum = LGAMMA_EXPANSION_COEF[7];

    for (int i = 6; i >= 0; i--) {
      sum *= z;
      sum += LGAMMA_EXPANSION_COEF[i];
    }
    float series = sum / abs_x;

    rc = (abs_x - 0.5) * ::metal::log(abs_x) - abs_x + HALF_LOG_TWO_PI + series;
  }

  if (x >= 0) {
    return rc;
  }

  // Reflection formula
  // Compute arg first to workaround Metal compiler bgg of sorts on M4
  // See https://github.com/pytorch/pytorch/pull/145740 for more details
  auto log_arg = abs_x * ::metal::abs(::metal::sinpi(abs_x));
  return LOG_PI - rc - ::metal::log(log_arg);
}

inline float zeta(float x, float q) {
  constexpr float MACHEP = 1.11022302462515654042E-16;
  constexpr float ZETA_EXPANSION[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9,
      7.47242496e10,
      -2.950130727918164224e12,
      1.1646782814350067249e14,
      -4.5979787224074726105e15,
      1.8152105401943546773e17,
      -7.1661652561756670113e18};
  if (x == 1.0f) {
    return INFINITY;
  }

  if (x < 1.0f) {
    return NAN;
  }

  if (q <= 0.0f) {
    if (q == ::metal::trunc(q)) {
      return INFINITY;
    }
    if (x != ::metal::trunc(x)) {
      return NAN;
    }
  }

  float s = ::metal::pow(q, -x);
  float a = q;
  int i = 0;
  float b = 0.0f;
  while ((i < 9) || (a <= 9.0f)) {
    i += 1;
    a += 1.0f;
    b = ::metal::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return s;
    }
  }

  float w = a;
  s += b * w / (x - 1.0f);
  s -= 0.5f * b;
  a = 1.0f;
  float t;
  float k = 0.0f;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / ZETA_EXPANSION[i];
    s += t;
    t = ::metal::fabs(t / s);
    if (t < MACHEP) {
      return s;
    }
    k += 1.0f;
    a *= x + k;
    b /= w;
    k += 1.0f;
  }
  return s;
}

template <typename T0>
inline float polygamma(const int64_t order, const T0 input) {
  float x = input;
  float n = order;
  float sgn = ((order % 2) ? 1 : -1);
  return sgn * gamma(n + 1) * zeta(n + 1, x);
}

inline float calc_digamma_positive_domain(float x) {
  constexpr float DIGAMMA_COEF[7] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  // Push x to be >= 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    constexpr float PSI_10 = 2.25175258906672110764;
    return result + PSI_10;
  }

  // Compute asymptotic digamma
  float y = 0;
  if (x < 1.0E+17) {
    float z = 1.0 / (x * x);
    for (int i = 0; i <= 6; i++) {
      y += ::metal::pow(z, i) * DIGAMMA_COEF[i];
    }
    y *= z;
  }
  return result + ::metal::log(x) - (0.5 / x) - y;
}

template <typename T0>
inline float digamma(T0 x) {
  if (x < 0.0f) {
    if (x == ::metal::trunc(x)) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return NAN;
    } else {
      // Extracts the fractional part of x as r, since tan(pi * r) is more
      // numerically accurate than tan(pi * x). While these operations are
      // mathematically equivalent since both x and r are in radians and tan()
      // has a periodicity of pi, in practice the computation of pi * x is a
      // source of error (when |x| > 1).
      float r = ::metal::fract(x);
      return calc_digamma_positive_domain(1.0f - x) -
          M_PI_F / ::metal::tan(M_PI_F * r);
    }
  } else if (x == 0.0f) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return ::metal::copysign(INFINITY, static_cast<float>(-x));
  } else {
    return calc_digamma_positive_domain(x);
  }
}

template <typename T>
inline ::metal::enable_if_t<is_scalar_floating_point_v<T>, T> sinc(T a) {
  if (a == static_cast<T>(0)) {
    return static_cast<T>(1);
  }
  auto product = M_PI_F * static_cast<float>(a);
  return static_cast<T>(::metal::precise::sin(product) / product);
}

// Complex sinc2 implementation
template <typename T>
inline ::metal::enable_if_t<is_complex_v<T>, T> sinc(T inp) {
  auto a = static_cast<float2>(inp) * M_PI_F;
  const float a2 = a.x * a.x + a.y * a.y;
  if (a2 == 0) {
    return 0;
  }
  float cosx;
  float sinx = ::metal::sincos(a.x, cosx);
  float sinhy = ::metal::sinh(a.y);
  float coshy = ::metal::cosh(a.y);
  auto re = sinx * coshy * a.x + cosx * sinhy * a.y;
  auto im = cosx * sinhy * a.x - sinx * coshy * a.y;
  return T(re, im) / a2;
}

template <typename T>
inline T spherical_bessel_j0(T x) {
  if (::metal::isinf(x))
    return T(0.0);
  T x2 = x * x;
  T k1 = static_cast<T>(-1.0);
  T k2 = static_cast<T>(1.0);

  if (::metal::fabs(static_cast<T>(x)) < T(0.5)) {
    return T(1.0) +
        x2 *
        (k1 / T(6.0) +
         x2 *
             (k2 / T(120.0) +
              x2 *
                  (k1 / T(5040.0) +
                   x2 *
                       (k2 / T(362880.0) +
                        x2 *
                            (k1 / T(39916800.0) +
                             x2 * (k2 / T(6227020800.0)))))));
  }

  return static_cast<T>(::metal::sin(x) / x);
}

// Compute log(1+x) without losing precision for small values of x
// Adapted from https://www.johndcook.com/blog/cpp_log_one_plus_x/
template <typename T>
inline float log1p(T x) {
  // x is large enough that the obvious evaluation is OK
  if (::metal::fabs(x) > 1E-4) {
    return ::metal::log(1. + x);
  }

  // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3
  // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
  return (-0.5 * x + 1.0) * x;
}

template <typename T>
inline float xlog1py(T x, T y) {
  if (::metal::isnan(y)) {
    return NAN;
  }

  if (x == 0) {
    return x;
  }

  return x * log1p(y);
}

template <typename T>
inline T entr(T a) {
  if (a != a) {
    return a;
  }

  if (a > 0) {
    return static_cast<T>(-a * ::metal::log(a));
  }

  if (a == 0) {
    return 0;
  }

  return static_cast<T>(-INFINITY);
}

} // namespace metal
} // namespace c10
