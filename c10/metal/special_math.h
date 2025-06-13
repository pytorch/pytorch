// Implementation of specal math functions for Metal
#pragma once
#include <c10/metal/expm1f.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

namespace c10 {
namespace metal {

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 * Copy-n-pasted from
 * https://github.com/ml-explore/mlx/blob/2e8cf0b4506c200a5c2d199ecbbf655fdf4c2ce2/mlx/backend/metal/kernels/erf.h#L11
 */
template <typename T>
inline float erf(T x) {
  const auto a = static_cast<float>(x);
  const auto t = ::metal::abs(a);
  const auto s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    auto r = ::metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    const auto u = ::metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = ::metal::fma(r, s, u);
    r = ::metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = ::metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = ::metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = ::metal::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - ::metal::exp(r);
    r = ::metal::copysign(r, a);
    return r;
  }

  // maximum error 0.98929 ulp
  auto r = -5.96761703e-4f; // -0x1.38e000p-11
  r = ::metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
  r = ::metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
  r = ::metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
  r = ::metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
  r = ::metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
  r = ::metal::fma(r, a, a);
  return r;
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

template <typename T>
inline T i0e(T _x) {
  auto x = ::metal::fabs(_x);

  if (x <= 8.0) {
    constexpr float coefficients[] = {
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
    return static_cast<T>(chbevl(y, coefficients, int{30}));
  }

  // x > 8
  constexpr float coefficients[] = {
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
      chbevl(32.0 / x - 2.0, coefficients, 25) / ::metal::sqrt(x));
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

template <typename T>
inline T i1e(T _x) {
  const auto x = ::metal::fabs(_x);
  if (x <= 8.0) {
    // Chebyshev double coefficients for exp(-x) i1(x) in the interval [0,8].
    // Note: lim(x->0){ exp(-x) i1(x) / x } = 1/2.
    constexpr float coefficients[] = {
        9.38153738649577178388E-9f,
        -4.44505912879632808065E-8f,
        2.00329475355213526229E-7f,
        -8.56872026469545474066E-7f,
        3.47025130813767847674E-6f,
        -1.32731636560394358279E-5f,
        4.78156510755005422638E-5f,
        -1.61760815825896745588E-4f,
        5.12285956168575772895E-4f,
        -1.51357245063125314899E-3f,
        4.15642294431288815669E-3f,
        -1.05640848946261981558E-2f,
        2.47264490306265168283E-2f,
        -5.29459812080949914269E-2f,
        1.02643658689847095384E-1f,
        -1.76416518357834055153E-1f,
        2.52587186443633654823E-1f};
    const auto y = x / 2.0 - 2.0;
    const auto out = chbevl(y, coefficients, 17) * x;
    return static_cast<T>(_x < 0. ? -out : out);
  }

  // Chebyshev coefficients for exp(-x) sqrt(x) i1(x)
  //   in the inverted interval (8, infinity].
  // Note: lim(x->inf){ exp(-x) sqrt(x) i1(x) } = 1/sqrt(2pi).
  // TODO: what's an "inverted interval"? Open on the left
  //   and closed on the right?
  constexpr float coefficients[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  const auto out =
      chbevl(32. / x - 2., coefficients, 7) / ::metal::precise::sqrt(x);
  return static_cast<T>(_x < 0. ? -out : out);
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

template <typename T0>
inline float polygamma(const int64_t order, const T0 input) {
  // Filter out n == 0.
  if (order == 0) {
    return digamma(input);
  }

  float x = input;
  float n = order;
  float sgn = ((order % 2) ? 1 : -1);
  return sgn * gamma(n + 1) * zeta(n + 1, x);
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

template <typename T>
inline float xlog1py(T x, T y) {
  if (::metal::isnan(y)) {
    return NAN;
  }

  if (x == 0) {
    return x;
  }

  return x * ::c10::metal::log1p(y);
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

// Copy-n-paste from aten/src/ATen/native/cuda/Math.cuh lines 1463-1915
template <typename T>
inline float bessel_j0_forward(T x) {
  constexpr float PP[] = {
      +7.96936729297347051624e-04,
      +8.28352392107440799803e-02,
      +1.23953371646414299388e+00,
      +5.44725003058768775090e+00,
      +8.74716500199817011941e+00,
      +5.30324038235394892183e+00,
      +9.99999999999999997821e-01,
  };

  constexpr float PQ[] = {
      +9.24408810558863637013e-04,
      +8.56288474354474431428e-02,
      +1.25352743901058953537e+00,
      +5.47097740330417105182e+00,
      +8.76190883237069594232e+00,
      +5.30605288235394617618e+00,
      +1.00000000000000000218e+00,
  };

  constexpr float QP[] = {
      -1.13663838898469149931e-02,
      -1.28252718670509318512e+00,
      -1.95539544257735972385e+01,
      -9.32060152123768231369e+01,
      -1.77681167980488050595e+02,
      -1.47077505154951170175e+02,
      -5.14105326766599330220e+01,
      -6.05014350600728481186e+00,
  };

  constexpr float QQ[] = {
      +6.43178256118178023184e+01,
      +8.56430025976980587198e+02,
      +3.88240183605401609683e+03,
      +7.24046774195652478189e+03,
      +5.93072701187316984827e+03,
      +2.06209331660327847417e+03,
      +2.42005740240291393179e+02,
  };

  constexpr float RP[] = {
      -4.79443220978201773821e+09,
      +1.95617491946556577543e+12,
      -2.49248344360967716204e+14,
      +9.70862251047306323952e+15,
  };

  constexpr float RQ[] = {
      +4.99563147152651017219e+02,
      +1.73785401676374683123e+05,
      +4.84409658339962045305e+07,
      +1.11855537045356834862e+10,
      +2.11277520115489217587e+12,
      +3.10518229857422583814e+14,
      +3.18121955943204943306e+16,
      +1.71086294081043136091e+18,
  };

  if (x < T(0)) {
    x = -x;
  }

  if (x <= T(5.0)) {
    if (x < T(0.00001)) {
      return 1.0 - x * x / 4.0;
    }

    float rp = 0.0;

    for (auto index = 0; index <= 3; index++) {
      rp = rp * (x * x) + RP[index];
    }

    float rq = 0.0;

    for (auto index = 0; index <= 7; index++) {
      rq = rq * (x * x) + RQ[index];
    }

    return (x * x - 5.78318596294678452118e+00) *
        (x * x - T(3.04712623436620863991e+01)) * rp / rq;
  }

  float pp = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pp = pp * (25.0 / (x * x)) + PP[index];
  }

  float pq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pq = pq * (25.0 / (x * x)) + PQ[index];
  }

  float qp = 0.0;

  for (auto index = 0; index <= 7; index++) {
    qp = qp * (25.0 / (x * x)) + QP[index];
  }

  float qq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    qq = qq * (25.0 / (x * x)) + QQ[index];
  }

  return (pp / pq *
              ::metal::precise::cos(
                  x - T(0.785398163397448309615660845819875721)) -
          5.0 / x * (qp / qq) *
              ::metal::precise::sin(
                  x - 0.785398163397448309615660845819875721)) *
      0.797884560802865355879892119868763737 / ::metal::precise::sqrt(x);
} // bessel_j0_forward(T x)

template <typename T>
inline float bessel_y0_forward(T x) {
  constexpr float PP[] = {
      +7.96936729297347051624e-04,
      +8.28352392107440799803e-02,
      +1.23953371646414299388e+00,
      +5.44725003058768775090e+00,
      +8.74716500199817011941e+00,
      +5.30324038235394892183e+00,
      +9.99999999999999997821e-01,
  };

  constexpr float PQ[] = {
      +9.24408810558863637013e-04,
      +8.56288474354474431428e-02,
      +1.25352743901058953537e+00,
      +5.47097740330417105182e+00,
      +8.76190883237069594232e+00,
      +5.30605288235394617618e+00,
      +1.00000000000000000218e+00,
  };

  constexpr float QP[] = {
      -1.13663838898469149931e-02,
      -1.28252718670509318512e+00,
      -1.95539544257735972385e+01,
      -9.32060152123768231369e+01,
      -1.77681167980488050595e+02,
      -1.47077505154951170175e+02,
      -5.14105326766599330220e+01,
      -6.05014350600728481186e+00,
  };

  constexpr float QQ[] = {
      +6.43178256118178023184e+01,
      +8.56430025976980587198e+02,
      +3.88240183605401609683e+03,
      +7.24046774195652478189e+03,
      +5.93072701187316984827e+03,
      +2.06209331660327847417e+03,
      +2.42005740240291393179e+02,
  };

  constexpr float YP[] = {
      +1.55924367855235737965e+04,
      -1.46639295903971606143e+07,
      +5.43526477051876500413e+09,
      -9.82136065717911466409e+11,
      +8.75906394395366999549e+13,
      -3.46628303384729719441e+15,
      +4.42733268572569800351e+16,
      -1.84950800436986690637e+16,
  };

  constexpr float YQ[] = {
      +1.04128353664259848412e+03,
      +6.26107330137134956842e+05,
      +2.68919633393814121987e+08,
      +8.64002487103935000337e+10,
      +2.02979612750105546709e+13,
      +3.17157752842975028269e+15,
      +2.50596256172653059228e+17,
  };

  if (x <= T(5.0)) {
    if (x == T(0.0)) {
      return -INFINITY;
    }

    if (x < T(0.0)) {
      return NAN;
    }

    float yp = 0.0;

    for (auto index = 0; index <= 7; index++) {
      yp = yp * (x * x) + YP[index];
    }

    float yq = 0.0;

    for (auto index = 0; index <= 6; index++) {
      yq = yq * (x * x) + YQ[index];
    }

    return yp / yq +
        (0.636619772367581343075535053490057448 * ::metal::precise::log(x) *
         bessel_j0_forward(x));
  }

  float pp = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pp = pp * (25.0 / (x * x)) + PP[index];
  }

  float pq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pq = pq * (25.0 / (x * x)) + PQ[index];
  }

  float qp = 0.0;

  for (auto index = 0; index <= 7; index++) {
    qp = qp * (25.0 / (x * x)) + QP[index];
  }

  float qq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    qq = qq * (25.0 / (x * x)) + QQ[index];
  }

  return (pp / pq *
              ::metal::precise::sin(
                  x - 0.785398163397448309615660845819875721) +
          5.0 / x * (qp / qq) *
              ::metal::precise::cos(
                  x - 0.785398163397448309615660845819875721)) *
      0.797884560802865355879892119868763737 / ::metal::precise::sqrt(x);
} // bessel_y0_forward(T x)

template <typename T>
inline float bessel_j1_forward(T x) {
  constexpr float PP[] = {
      +7.62125616208173112003e-04,
      +7.31397056940917570436e-02,
      +1.12719608129684925192e+00,
      +5.11207951146807644818e+00,
      +8.42404590141772420927e+00,
      +5.21451598682361504063e+00,
      +1.00000000000000000254e+00,
  };

  constexpr float PQ[] = {
      +5.71323128072548699714e-04,
      +6.88455908754495404082e-02,
      +1.10514232634061696926e+00,
      +5.07386386128601488557e+00,
      +8.39985554327604159757e+00,
      +5.20982848682361821619e+00,
      +9.99999999999999997461e-01,
  };

  constexpr float QP[] = {
      +5.10862594750176621635e-02,
      +4.98213872951233449420e+00,
      +7.58238284132545283818e+01,
      +3.66779609360150777800e+02,
      +7.10856304998926107277e+02,
      +5.97489612400613639965e+02,
      +2.11688757100572135698e+02,
      +2.52070205858023719784e+01,
  };

  constexpr float QQ[] = {
      +7.42373277035675149943e+01,
      +1.05644886038262816351e+03,
      +4.98641058337653607651e+03,
      +9.56231892404756170795e+03,
      +7.99704160447350683650e+03,
      +2.82619278517639096600e+03,
      +3.36093607810698293419e+02,
  };

  constexpr float RP[] = {
      -8.99971225705559398224e+08,
      +4.52228297998194034323e+11,
      -7.27494245221818276015e+13,
      +3.68295732863852883286e+15,
  };

  constexpr float RQ[] = {
      +6.20836478118054335476e+02,
      +2.56987256757748830383e+05,
      +8.35146791431949253037e+07,
      +2.21511595479792499675e+10,
      +4.74914122079991414898e+12,
      +7.84369607876235854894e+14,
      +8.95222336184627338078e+16,
      +5.32278620332680085395e+18,
  };

  if (x < T(0.0)) {
    return -bessel_j1_forward(-x);
  }

  if (x <= T(5.0)) {
    float rp = 0.0;

    for (auto index = 0; index <= 3; index++) {
      rp = rp * (x * x) + RP[index];
    }

    float rq = 0.0;

    for (auto index = 0; index <= 7; index++) {
      rq = rq * (x * x) + RQ[index];
    }

    return rp / rq * x * (x * x - 1.46819706421238932572e+01) *
        (x * x - 4.92184563216946036703e+01);
  }

  float pp = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pp = pp * (5.0 / x * (5.0 / x)) + PP[index];
  }

  float pq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pq = pq * (5.0 / x * (5.0 / x)) + PQ[index];
  }

  float qp = 0.0;

  for (auto index = 0; index <= 7; index++) {
    qp = qp * (5.0 / x * (5.0 / x)) + QP[index];
  }

  float qq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    qq = qq * (5.0 / x * (5.0 / x)) + QQ[index];
  }

  return (pp / pq *
              ::metal::precise::cos(
                  x - 2.356194490192344928846982537459627163) -
          5.0 / x * (qp / qq) *
              ::metal::precise::sin(
                  x - 2.356194490192344928846982537459627163)) *
      0.797884560802865355879892119868763737 / ::metal::precise::sqrt(x);
} // bessel_j1_forward(T x)

template <typename T>
inline float bessel_y1_forward(T x) {
  constexpr float PP[] = {
      +7.62125616208173112003e-04,
      +7.31397056940917570436e-02,
      +1.12719608129684925192e+00,
      +5.11207951146807644818e+00,
      +8.42404590141772420927e+00,
      +5.21451598682361504063e+00,
      +1.00000000000000000254e+00,
  };

  constexpr float PQ[] = {
      +5.71323128072548699714e-04,
      +6.88455908754495404082e-02,
      +1.10514232634061696926e+00,
      +5.07386386128601488557e+00,
      +8.39985554327604159757e+00,
      +5.20982848682361821619e+00,
      +9.99999999999999997461e-01,
  };

  constexpr float QP[] = {
      +5.10862594750176621635e-02,
      +4.98213872951233449420e+00,
      +7.58238284132545283818e+01,
      +3.66779609360150777800e+02,
      +7.10856304998926107277e+02,
      +5.97489612400613639965e+02,
      +2.11688757100572135698e+02,
      +2.52070205858023719784e+01,
  };

  constexpr float QQ[] = {
      +7.42373277035675149943e+01,
      +1.05644886038262816351e+03,
      +4.98641058337653607651e+03,
      +9.56231892404756170795e+03,
      +7.99704160447350683650e+03,
      +2.82619278517639096600e+03,
      +3.36093607810698293419e+02,
  };

  constexpr float YP[] = {
      +1.26320474790178026440e+09,
      -6.47355876379160291031e+11,
      +1.14509511541823727583e+14,
      -8.12770255501325109621e+15,
      +2.02439475713594898196e+17,
      -7.78877196265950026825e+17,
  };

  constexpr float YQ[] = {
      +5.94301592346128195359e+02,
      +2.35564092943068577943e+05,
      +7.34811944459721705660e+07,
      +1.87601316108706159478e+10,
      +3.88231277496238566008e+12,
      +6.20557727146953693363e+14,
      +6.87141087355300489866e+16,
      +3.97270608116560655612e+18,
  };

  if (x <= T(5.0)) {
    if (x == T(0.0)) {
      return -INFINITY;
    }

    if (x <= T(0.0)) {
      return NAN;
    }

    float yp = 0.0;

    for (auto index = 0; index <= 5; index++) {
      yp = yp * (x * x) + YP[index];
    }

    float yq = 0.0;

    for (auto index = 0; index <= 7; index++) {
      yq = yq * (x * x) + YQ[index];
    }

    return x * (yp / yq) +
        (0.636619772367581343075535053490057448 *
         (bessel_j1_forward(x) * ::metal::precise::log(x) - 1.0 / x));
  }

  float pp = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pp = pp * (5.0 / x * (5.0 / x)) + PP[index];
  }

  float pq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    pq = pq * (5.0 / x * (5.0 / x)) + PQ[index];
  }

  float qp = 0.0;

  for (auto index = 0; index <= 7; index++) {
    qp = qp * (5.0 / x * (5.0 / x)) + QP[index];
  }

  float qq = 0.0;

  for (auto index = 0; index <= 6; index++) {
    qq = qq * (5.0 / x * (5.0 / x)) + QQ[index];
  }

  return (pp / pq *
              ::metal::precise::sin(
                  x - 2.356194490192344928846982537459627163) +
          5.0 / x * (qp / qq) *
              ::metal::precise::cos(
                  x - 2.356194490192344928846982537459627163)) *
      0.797884560802865355879892119868763737 / ::metal::precise::sqrt(x);
} // bessel_y1_forward(T x)

template <typename T>
inline float modified_bessel_i0_forward(T x) {
  constexpr float A[] = {
      -4.41534164647933937950e-18, +3.33079451882223809783e-17,
      -2.43127984654795469359e-16, +1.71539128555513303061e-15,
      -1.16853328779934516808e-14, +7.67618549860493561688e-14,
      -4.85644678311192946090e-13, +2.95505266312963983461e-12,
      -1.72682629144155570723e-11, +9.67580903537323691224e-11,
      -5.18979560163526290666e-10, +2.65982372468238665035e-09,
      -1.30002500998624804212e-08, +6.04699502254191894932e-08,
      -2.67079385394061173391e-07, +1.11738753912010371815e-06,
      -4.41673835845875056359e-06, +1.64484480707288970893e-05,
      -5.75419501008210370398e-05, +1.88502885095841655729e-04,
      -5.76375574538582365885e-04, +1.63947561694133579842e-03,
      -4.32430999505057594430e-03, +1.05464603945949983183e-02,
      -2.37374148058994688156e-02, +4.93052842396707084878e-02,
      -9.49010970480476444210e-02, +1.71620901522208775349e-01,
      -3.04682672343198398683e-01, +6.76795274409476084995e-01,
  };

  constexpr float B[] = {
      -7.23318048787475395456e-18, -4.83050448594418207126e-18,
      +4.46562142029675999901e-17, +3.46122286769746109310e-17,
      -2.82762398051658348494e-16, -3.42548561967721913462e-16,
      +1.77256013305652638360e-15, +3.81168066935262242075e-15,
      -9.55484669882830764870e-15, -4.15056934728722208663e-14,
      +1.54008621752140982691e-14, +3.85277838274214270114e-13,
      +7.18012445138366623367e-13, -1.79417853150680611778e-12,
      -1.32158118404477131188e-11, -3.14991652796324136454e-11,
      +1.18891471078464383424e-11, +4.94060238822496958910e-10,
      +3.39623202570838634515e-09, +2.26666899049817806459e-08,
      +2.04891858946906374183e-07, +2.89137052083475648297e-06,
      +6.88975834691682398426e-05, +3.36911647825569408990e-03,
      +8.04490411014108831608e-01,
  };

  float p;
  float q = 0.0;

  if (::metal::fabs(x) <= 8.0) {
    float a = A[0];

    for (uint8_t index = 1; index < 30; index++) {
      p = q;
      q = a;
      a = (.5 * ::metal::fabs(x) - 2.0) * q - p + A[index];
    }

    return ::metal::exp(::metal::fabs(x)) * (T(0.5) * (a - p));
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (32.0 / ::metal::fabs(x) - 2.0) * q - p + B[index];
  }

  return ::metal::exp(::metal::fabs(x)) * (.5 * (b - p)) /
      ::metal::precise::sqrt(::metal::fabs(x));
} // modified_bessel_i0_forward(T x)

template <typename T>
inline float modified_bessel_i1_forward(T x) {
  constexpr float A[] = {
      +2.77791411276104639959e-18, -2.11142121435816608115e-17,
      +1.55363195773620046921e-16, -1.10559694773538630805e-15,
      +7.60068429473540693410e-15, -5.04218550472791168711e-14,
      +3.22379336594557470981e-13, -1.98397439776494371520e-12,
      +1.17361862988909016308e-11, -6.66348972350202774223e-11,
      +3.62559028155211703701e-10, -1.88724975172282928790e-09,
      +9.38153738649577178388e-09, -4.44505912879632808065e-08,
      +2.00329475355213526229e-07, -8.56872026469545474066e-07,
      +3.47025130813767847674e-06, -1.32731636560394358279e-05,
      +4.78156510755005422638e-05, -1.61760815825896745588e-04,
      +5.12285956168575772895e-04, -1.51357245063125314899e-03,
      +4.15642294431288815669e-03, -1.05640848946261981558e-02,
      +2.47264490306265168283e-02, -5.29459812080949914269e-02,
      +1.02643658689847095384e-01, -1.76416518357834055153e-01,
      +2.52587186443633654823e-01,
  };

  constexpr float B[] = {
      +7.51729631084210481353e-18, +4.41434832307170791151e-18,
      -4.65030536848935832153e-17, -3.20952592199342395980e-17,
      +2.96262899764595013876e-16, +3.30820231092092828324e-16,
      -1.88035477551078244854e-15, -3.81440307243700780478e-15,
      +1.04202769841288027642e-14, +4.27244001671195135429e-14,
      -2.10154184277266431302e-14, -4.08355111109219731823e-13,
      -7.19855177624590851209e-13, +2.03562854414708950722e-12,
      +1.41258074366137813316e-11, +3.25260358301548823856e-11,
      -1.89749581235054123450e-11, -5.58974346219658380687e-10,
      -3.83538038596423702205e-09, -2.63146884688951950684e-08,
      -2.51223623787020892529e-07, -3.88256480887769039346e-06,
      -1.10588938762623716291e-04, -9.76109749136146840777e-03,
      +7.78576235018280120474e-01,
  };

  float p;
  float q = 0.0;

  if (::metal::fabs(x) <= T(8.0)) {
    float a = A[0];

    for (uint8_t index = 1; index < 29; index++) {
      p = q;
      q = a;
      a = (.5 * ::metal::fabs(x) - 2.0) * q - p + A[index];
    }

    return .5 * (a - p) * x * ::metal::precise::exp(::metal::fabs(x));
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (32.0 / ::metal::fabs(x) - 2.0) * q - p + B[index];
  }

  if (x < 0.0) {
    return -(
        ::metal::precise::exp(::metal::fabs(x)) * (0.5 * (b - p)) /
        ::metal::precise::sqrt(::metal::fabs(x)));
  }

  return ::metal::precise::exp(::metal::fabs(x)) * (0.5 * (b - p)) /
      ::metal::precise::sqrt(::metal::fabs(x));
} // modified_bessel_i1_forward(T x)

template <typename T>
inline float modified_bessel_k0_forward(T x) {
  constexpr float A[] = {
      +1.37446543561352307156e-16,
      +4.25981614279661018399e-14,
      +1.03496952576338420167e-11,
      +1.90451637722020886025e-09,
      +2.53479107902614945675e-07,
      +2.28621210311945178607e-05,
      +1.26461541144692592338e-03,
      +3.59799365153615016266e-02,
      +3.44289899924628486886e-01,
      -5.35327393233902768720e-01,
  };

  constexpr float B[] = {
      +5.30043377268626276149e-18, -1.64758043015242134646e-17,
      +5.21039150503902756861e-17, -1.67823109680541210385e-16,
      +5.51205597852431940784e-16, -1.84859337734377901440e-15,
      +6.34007647740507060557e-15, -2.22751332699166985548e-14,
      +8.03289077536357521100e-14, -2.98009692317273043925e-13,
      +1.14034058820847496303e-12, -4.51459788337394416547e-12,
      +1.85594911495471785253e-11, -7.95748924447710747776e-11,
      +3.57739728140030116597e-10, -1.69753450938905987466e-09,
      +8.57403401741422608519e-09, -4.66048989768794782956e-08,
      +2.76681363944501510342e-07, -1.83175552271911948767e-06,
      +1.39498137188764993662e-05, -1.28495495816278026384e-04,
      +1.56988388573005337491e-03, -3.14481013119645005427e-02,
      +2.44030308206595545468e+00,
  };

  if (x == 0.0) {
    return INFINITY;
  }

  if (x < 0.0) {
    return NAN;
  }

  float p;
  float q = 0.0;

  if (x <= 2.0) {
    float a = A[0];

    for (uint8_t index = 1; index < 10; index++) {
      p = q;
      q = a;
      a = (x * x - 2.0) * q - p + A[index];
    }

    return 0.5 * (a - p) -
        ::metal::log(0.5 * x) * modified_bessel_i0_forward(x);
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (8.0 / x - 2.0) * q - p + B[index];
  }

  return ::metal::exp(-x) * (0.5 * (b - p)) / ::metal::sqrt(x);
} // modified_bessel_k0_forward(T x)

template <typename T>
inline float modified_bessel_k1_forward(T x) {
  constexpr float A[] = {
      -7.02386347938628759343e-18,
      -2.42744985051936593393e-15,
      -6.66690169419932900609e-13,
      -1.41148839263352776110e-10,
      -2.21338763073472585583e-08,
      -2.43340614156596823496e-06,
      -1.73028895751305206302e-04,
      -6.97572385963986435018e-03,
      -1.22611180822657148235e-01,
      -3.53155960776544875667e-01,
      +1.52530022733894777053e+00,
  };

  constexpr float B[] = {
      -5.75674448366501715755e-18, +1.79405087314755922667e-17,
      -5.68946255844285935196e-17, +1.83809354436663880070e-16,
      -6.05704724837331885336e-16, +2.03870316562433424052e-15,
      -7.01983709041831346144e-15, +2.47715442448130437068e-14,
      -8.97670518232499435011e-14, +3.34841966607842919884e-13,
      -1.28917396095102890680e-12, +5.13963967348173025100e-12,
      -2.12996783842756842877e-11, +9.21831518760500529508e-11,
      -4.19035475934189648750e-10, +2.01504975519703286596e-09,
      -1.03457624656780970260e-08, +5.74108412545004946722e-08,
      -3.50196060308781257119e-07, +2.40648494783721712015e-06,
      -1.93619797416608296024e-05, +1.95215518471351631108e-04,
      -2.85781685962277938680e-03, +1.03923736576817238437e-01,
      +2.72062619048444266945e+00,
  };

  if (x == 0.0) {
    return INFINITY;
  }

  if (x < 0.0) {
    return NAN;
  }

  float p;
  float q = 0.0;

  if (x <= 2.0) {
    float a = A[0];

    for (uint8_t index = 1; index < 11; index++) {
      p = q;
      q = a;
      a = (x * x - T(2.0)) * q - p + A[index];
    }

    return ::metal::precise::log(T(0.5) * x) * modified_bessel_i1_forward(x) +
        0.5 * (a - p) / x;
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (8.0 / x - 2.0) * q - p + B[index];
  }

  return ::metal::precise::exp(-x) * (0.5 * (b - p)) /
      ::metal::precise::sqrt(x);
}

template <typename T>
inline float scaled_modified_bessel_k0_forward(T x) {
  constexpr float A[] = {
      +1.37446543561352307156e-16,
      +4.25981614279661018399e-14,
      +1.03496952576338420167e-11,
      +1.90451637722020886025e-09,
      +2.53479107902614945675e-07,
      +2.28621210311945178607e-05,
      +1.26461541144692592338e-03,
      +3.59799365153615016266e-02,
      +3.44289899924628486886e-01,
      -5.35327393233902768720e-01,
  };

  constexpr float B[] = {
      +5.30043377268626276149e-18, -1.64758043015242134646e-17,
      +5.21039150503902756861e-17, -1.67823109680541210385e-16,
      +5.51205597852431940784e-16, -1.84859337734377901440e-15,
      +6.34007647740507060557e-15, -2.22751332699166985548e-14,
      +8.03289077536357521100e-14, -2.98009692317273043925e-13,
      +1.14034058820847496303e-12, -4.51459788337394416547e-12,
      +1.85594911495471785253e-11, -7.95748924447710747776e-11,
      +3.57739728140030116597e-10, -1.69753450938905987466e-09,
      +8.57403401741422608519e-09, -4.66048989768794782956e-08,
      +2.76681363944501510342e-07, -1.83175552271911948767e-06,
      +1.39498137188764993662e-05, -1.28495495816278026384e-04,
      +1.56988388573005337491e-03, -3.14481013119645005427e-02,
      +2.44030308206595545468e+00,
  };

  if (x == 0.0) {
    return INFINITY;
  }

  if (x < 0.0) {
    return NAN;
  }

  float p;
  float q = 0.0;

  if (x <= 2.0) {
    float a = A[0];

    for (uint8_t index = 1; index < 10; index++) {
      p = q;
      q = a;
      a = (x * x - T(2.0)) * q - p + A[index];
    }

    return (0.5 * (a - p) -
            ::metal::precise::log(0.5 * x) * modified_bessel_i0_forward(x)) *
        ::metal::precise::exp(x);
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (8.0 / x - 2.0) * q - p + B[index];
  }

  return 0.5 * (b - p) / ::metal::precise::sqrt(x);
}

template <typename T>
inline float scaled_modified_bessel_k1_forward(T x) {
  constexpr float A[] = {
      -7.02386347938628759343e-18,
      -2.42744985051936593393e-15,
      -6.66690169419932900609e-13,
      -1.41148839263352776110e-10,
      -2.21338763073472585583e-08,
      -2.43340614156596823496e-06,
      -1.73028895751305206302e-04,
      -6.97572385963986435018e-03,
      -1.22611180822657148235e-01,
      -3.53155960776544875667e-01,
      +1.52530022733894777053e+00,
  };

  constexpr float B[] = {
      -5.75674448366501715755e-18, +1.79405087314755922667e-17,
      -5.68946255844285935196e-17, +1.83809354436663880070e-16,
      -6.05704724837331885336e-16, +2.03870316562433424052e-15,
      -7.01983709041831346144e-15, +2.47715442448130437068e-14,
      -8.97670518232499435011e-14, +3.34841966607842919884e-13,
      -1.28917396095102890680e-12, +5.13963967348173025100e-12,
      -2.12996783842756842877e-11, +9.21831518760500529508e-11,
      -4.19035475934189648750e-10, +2.01504975519703286596e-09,
      -1.03457624656780970260e-08, +5.74108412545004946722e-08,
      -3.50196060308781257119e-07, +2.40648494783721712015e-06,
      -1.93619797416608296024e-05, +1.95215518471351631108e-04,
      -2.85781685962277938680e-03, +1.03923736576817238437e-01,
      +2.72062619048444266945e+00,
  };

  if (x == 0.0) {
    return INFINITY;
  }

  if (x < 0.0) {
    return NAN;
  }

  float p;
  float q = 0.0;

  if (x <= 2.0) {
    float a = A[0];

    for (uint8_t index = 1; index < 11; index++) {
      p = q;
      q = a;
      a = (x * x - 2.0) * q - p + A[index];
    }

    return (::metal::precise::log(0.5 * x) * modified_bessel_i1_forward(x) +
            0.5 * (a - p) / x) *
        ::metal::precise::exp(x);
  }

  float b = B[0];

  for (uint8_t index = 1; index < 25; index++) {
    p = q;
    q = b;
    b = (8.0 / x - 2.0) * q - p + B[index];
  }

  return (0.5 * (b - p) / ::metal::precise::sqrt(x));
}

template <typename T>
float chebyshev_polynomial_t_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (::metal::fabs(x) == 1.0) {
    if (x > 0.0 || n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  if ((n > 6) && (::metal::precise::fabs(x) < 1.0)) {
    return ::metal::precise::cos(n * ::metal::precise::acos(x));
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return x;
  }

  float p = 1.0;
  float q = x;
  float r;

  for (int64_t k = 2; k <= n; k++) {
    r = (x + x) * q - p;
    p = q;
    q = r;
  }
  return r;
}

template <typename T>
float chebyshev_polynomial_u_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (::metal::fabs(x) == 1.0) {
    if (x > 0.0 || n % 2 == 0) {
      return n + 1;
    }

    return -(n + 1);
  }

  if ((n > 8) && (::metal::fabs(x) < 1.0)) {
    const auto acos_x = ::metal::precise::acos(x);
    if (::metal::precise::sin(acos_x) != 0.0) {
      return ::metal::precise::sin((n + 1) * acos_x) /
          ::metal::precise::sin(acos_x);
    }

    return (n + 1) * ::metal::precise::cos((n + 1) * acos_x) / x;
  }

  if (n == 0) {
    return 1.0;
  }

  auto q = 2.0 * x;
  if (n == 1) {
    return q;
  }

  auto p = 1.0;
  float r;

  for (int64_t k = 2; k <= n; k++) {
    r = 2 * x * q - p;
    p = q;
    q = r;
  }

  return r;
}

template <typename T>
float chebyshev_polynomial_v_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (::metal::fabs(x) == 1.0) {
    if (x > 0.0) {
      return 1.0;
    }

    if (n % 2 == 0) {
      return n + n + 1;
    }

    return -(n + n + 1);
  }

  if ((n > 8) && (::metal::fabs(x) < 1.0)) {
    const auto acos_x = ::metal::precise::acos(x);
    if (::metal::precise::sin(.5 * acos_x) != 1.0) {
      return ::metal::precise::cos((n + 0.5) * acos_x) /
          ::metal::precise::cos(.5 * acos_x);
    }

    if (n % 2 == 0) {
      return n + n + 1;
    }

    return -(n + n + 1);
  }

  if (n == 0) {
    return 1.0;
  }

  auto q = 2.0 * x - 1.0;
  if (n == 1) {
    return q;
  }

  auto p = 1.0;
  float r;

  for (int64_t k = 2; k <= n; k++) {
    r = 2 * x * q - p;
    p = q;
    q = r;
  }

  return r;
} // chebyshev_polynomial_v_forward(T x, int64_t n)

template <typename T>
float chebyshev_polynomial_w_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (::metal::fabs(x) == 1.0) {
    if (x > 0.0) {
      return n + n + 1;
    }

    if (n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  if ((n > 8) && (::metal::fabs(x) < 1.0)) {
    const auto acos_x = ::metal::precise::acos(x);
    if (::metal::precise::cos(.5 * acos_x) != 1.0) {
      return ::metal::precise::sin((n + 0.5) * acos_x) /
          ::metal::precise::sin(.5 * acos_x);
    }

    if (x > 0.0) {
      return n + n + 1;
    }

    if (n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  if (n == 0) {
    return 1.0;
  }

  auto q = 2.0 * x + 1.0;
  if (n == 1) {
    return q;
  }

  auto p = 1.0;
  float r;

  for (int64_t k = 2; k <= n; k++) {
    r = 2.0 * x * q - p;
    p = q;
    q = r;
  }

  return r;
} // chebyshev_polynomial_w_forward(T x, int64_t n)

template <typename T>
// TODO: Add 512 if/when double will be supported in Metal
inline constexpr int getHermitianLimit() {
  return 128;
}

template <typename T>
inline float hermite_polynomial_h_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return x + x;
  }

  if (n > getHermitianLimit<T>()) {
    return NAN;
  }

  float p = 1.0;
  float q = x + x;
  float r = 0.0;

  for (int64_t k = 2; k < n + n; k += 2) {
    r = (x + x) * q - k * p;
    p = q;
    q = r;
  }

  return r;
} // hermite_polynomial_h_forward(T x, int64_t n)

template <typename T>
inline float hermite_polynomial_he_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return x;
  }

  if (n > getHermitianLimit<T>()) {
    return NAN;
  }

  float p = 1.0;
  float q = x;
  float r;

  for (int64_t k = 1; k < n; k++) {
    r = x * q - k * p;
    p = q;
    q = r;
  }

  return r;
} // hermite_polynomial_he_forward(T x, int64_t n)

} // namespace metal
} // namespace c10
