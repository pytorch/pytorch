// Implementation of special math functions for Metal
#pragma once
#include <c10/metal/expm1f.h>
#include <c10/metal/igamma.h>
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
float erfc(T x) {
  return 1.0 - erf(x);
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
inline ::metal::enable_if_t<is_scalar_floating_point_v<T>, T> logaddexp(
    T a,
    T b) {
  float a0 = static_cast<float>(a);
  float b0 = static_cast<float>(b);
  if (::metal::isinf(a0) && a0 == b0) {
    return static_cast<T>(a0);
  } else {
    float m0 = ::metal::max(a0, b0);
    return static_cast<T>(
        m0 + ::c10::metal::log1p(::metal::exp(-::metal::abs(a0 - b0))));
  }
}

// The function is ported from mlx
template <typename T>
inline ::metal::enable_if_t<is_complex_v<T>, T> logaddexp(T a, T b) {
  if (::metal::isnan(a.x) || ::metal::isnan(a.y) || ::metal::isnan(b.x) ||
      ::metal::isnan(b.y)) {
    return T(NAN, NAN);
  }

  T maxval = a.x > b.x ? a : b;
  T minval = a.x < b.x ? a : b;
  constexpr auto inf = ::metal::numeric_limits<T>::infinity().x;

  if (minval.x == -inf || maxval.x == inf) {
    return maxval;
  }

  float2 maxval_ = static_cast<float2>(maxval);
  float2 minval_ = static_cast<float2>(minval);
  float m = ::metal::exp(minval_.x - maxval_.x);
  float2 dexp{
      m * ::metal::cos(minval_.y - maxval_.y),
      m * ::metal::sin(minval_.y - maxval_.y),
  };
  return static_cast<T>(maxval_ + ::c10::metal::log1p(dexp));
}

template <typename T>
inline T logaddexp2(T a, T b) {
  constexpr auto log_2 = float(0.693147180559945309417232121458176);
  constexpr auto inv_log_2 = float(1) / log_2;
  float a0 = static_cast<float>(a);
  float b0 = static_cast<float>(b);
  if (::metal::isinf(a0) && a0 == b0) {
    return static_cast<T>(a0);
  } else {
    float m0 = ::metal::max(a0, b0);
    return static_cast<T>(
        m0 +
        ::c10::metal::log1p(::metal::pow(float(2), -::metal::abs(a0 - b0))) *
            inv_log_2);
  }
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

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
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

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
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

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
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

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
    r = 2.0 * x * q - p;
    p = q;
    q = r;
  }

  return r;
} // chebyshev_polynomial_w_forward(T x, int64_t n)

template <typename T>
float shifted_chebyshev_polynomial_t_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (x == T(1.0)) {
    return 1.0;
  }

  if (x == 0.0) {
    if (n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  const float xpxm1 = x + x - 1.0;
  if ((n > 6) && (::metal::abs(xpxm1) < 1.0)) {
    return ::metal::precise::cos(n * ::metal::precise::acos(xpxm1));
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return xpxm1;
  }

  float p = 1.0;
  float q = xpxm1;
  float r;

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
    r = (xpxm1 + xpxm1) * q - p;
    p = q;
    q = r;
  }

  return r;
} // shifted_chebyshev_polynomial_t_forward(T x, int64_t n)

template <typename T>
float shifted_chebyshev_polynomial_u_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (x == 1.0) {
    return n + 1;
  }

  if (x == 0.0) {
    if (n % 2 == 0) {
      return n + 1;
    }

    return -(n + 1);
  }
  const float xpxm1 = x + x - 1.0;
  if ((n > 6) && (::metal::abs(xpxm1) < 1.0)) {
    const float acos_2xm1 = ::metal::precise::acos(xpxm1);
    const float divisor = ::metal::precise::sin(acos_2xm1);
    if (divisor != 0.0) {
      return ::metal::precise::sin((n + 1) * acos_2xm1) / divisor;
    }

    return (n + 1) * ::metal::precise::cos((n + 1) * acos_2xm1) / xpxm1;
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return xpxm1 + xpxm1;
  }

  float p = 1.0;
  float q = xpxm1 + xpxm1;
  float r;

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
    r = (xpxm1 + xpxm1) * q - p;
    p = q;
    q = r;
  }

  return r;
} // shifted_chebyshev_polynomial_u_forward(T x, int64_t n)

template <typename T>
float shifted_chebyshev_polynomial_v_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (x == 1.0) {
    return 1.0;
  }

  if (x == 0.0) {
    if (n % 2 == 0) {
      return (n + n + 1);
    }

    return -(n + n + 1);
  }

  const float xpxm1 = x + x - 1.0;
  if ((n > 6) && (::metal::abs(xpxm1) < 1.0)) {
    const float acos_2xm1 = ::metal::precise::acos(xpxm1);
    if (::metal::precise::sin(acos_2xm1 / 2.0) != 1.0) {
      return ::metal::precise::cos((n + 0.5) * acos_2xm1) /
          ::metal::precise::cos(acos_2xm1 / 2.0);
    }

    if (n % 2 == 0) {
      return n + n + 1;
    }

    return -(n + n + 1);
  }

  if (n == 0) {
    return T(1.0);
  }

  if (n == 1) {
    return xpxm1 + xpxm1 - 1.0;
  }

  float p = 1.0;
  float q = xpxm1 + xpxm1 - 1.0;
  float r;

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
    r = (xpxm1 + xpxm1) * q - p;
    p = q;
    q = r;
  }

  return r;
} // shifted_chebyshev_polynomial_v_forward(T x, int64_t n)

template <typename T>
float shifted_chebyshev_polynomial_w_forward(T x, int64_t n) {
  if (n < 0) {
    return 0.0;
  }

  if (x == 1.0) {
    return n + n + 1;
  }

  if (x == 0.0) {
    if (n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  const float xpxm1 = x + x - 1.0;
  if ((n > 4) && (::metal::abs(xpxm1) < 1.0)) {
    const float acos_2xm1 = ::metal::precise::acos(xpxm1);
    if (::metal::precise::cos(acos_2xm1 / 2.0) != 1.0) {
      return ::metal::precise::sin((n + 0.5) * acos_2xm1) /
          ::metal::precise::sin(acos_2xm1 / 2.0);
    }

    if (n % 2 == 0) {
      return 1.0;
    }

    return -1.0;
  }

  if (n == 0) {
    return 1.0;
  }

  if (n == 1) {
    return xpxm1 + xpxm1 + 1.0;
  }

  float p = 1.0;
  float q = xpxm1 + xpxm1 + 1.0;
  float r;

  for (int64_t k = 2; (k <= n) && !::metal::isnan(q); k++) {
    r = (xpxm1 + xpxm1) * q - p;
    p = q;
    q = r;
  }

  return r;
} // shifted_chebyshev_polynomial_w_forward(T x, int64_t n)

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

/* The next function is taken from http://ab-initio.mit.edu/faddeeva */

/* Copyright (c) 2012 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define IMPL_ERFCX_Y100_CASE(X, A, B, C, D, E, F, G)                    \
  case X: {                                                             \
    const auto t = 2.0 * y100 - (2 * X + 1);                            \
    return A + (B + (C + (D + (E + (F + G * t) * t) * t) * t) * t) * t; \
  }

template <typename T>
inline T erfcx_y100(T y100) {
  switch (static_cast<int>(y100)) {
    IMPL_ERFCX_Y100_CASE(
        0,
        0.70878032454106438663e-3,
        0.71234091047026302958e-3,
        0.35779077297597742384e-5,
        0.17403143962587937815e-7,
        0.81710660047307788845e-10,
        0.36885022360434957634e-12,
        0.15917038551111111111e-14)
    IMPL_ERFCX_Y100_CASE(
        1,
        0.21479143208285144230e-2,
        0.72686402367379996033e-3,
        0.36843175430938995552e-5,
        0.18071841272149201685e-7,
        0.85496449296040325555e-10,
        0.38852037518534291510e-12,
        0.16868473576888888889e-14)
    IMPL_ERFCX_Y100_CASE(
        2,
        0.36165255935630175090e-2,
        0.74182092323555510862e-3,
        0.37948319957528242260e-5,
        0.18771627021793087350e-7,
        0.89484715122415089123e-10,
        0.40935858517772440862e-12,
        0.17872061464888888889e-14)
    IMPL_ERFCX_Y100_CASE(
        3,
        0.51154983860031979264e-2,
        0.75722840734791660540e-3,
        0.39096425726735703941e-5,
        0.19504168704300468210e-7,
        0.93687503063178993915e-10,
        0.43143925959079664747e-12,
        0.18939926435555555556e-14)
    IMPL_ERFCX_Y100_CASE(
        4,
        0.66457513172673049824e-2,
        0.77310406054447454920e-3,
        0.40289510589399439385e-5,
        0.20271233238288381092e-7,
        0.98117631321709100264e-10,
        0.45484207406017752971e-12,
        0.20076352213333333333e-14)
    IMPL_ERFCX_Y100_CASE(
        5,
        0.82082389970241207883e-2,
        0.78946629611881710721e-3,
        0.41529701552622656574e-5,
        0.21074693344544655714e-7,
        0.10278874108587317989e-9,
        0.47965201390613339638e-12,
        0.21285907413333333333e-14)
    IMPL_ERFCX_Y100_CASE(
        6,
        0.98039537275352193165e-2,
        0.80633440108342840956e-3,
        0.42819241329736982942e-5,
        0.21916534346907168612e-7,
        0.10771535136565470914e-9,
        0.50595972623692822410e-12,
        0.22573462684444444444e-14)
    IMPL_ERFCX_Y100_CASE(
        7,
        0.11433927298290302370e-1,
        0.82372858383196561209e-3,
        0.44160495311765438816e-5,
        0.22798861426211986056e-7,
        0.11291291745879239736e-9,
        0.53386189365816880454e-12,
        0.23944209546666666667e-14)
    IMPL_ERFCX_Y100_CASE(
        8,
        0.13099232878814653979e-1,
        0.84167002467906968214e-3,
        0.45555958988457506002e-5,
        0.23723907357214175198e-7,
        0.11839789326602695603e-9,
        0.56346163067550237877e-12,
        0.25403679644444444444e-14)
    IMPL_ERFCX_Y100_CASE(
        9,
        0.14800987015587535621e-1,
        0.86018092946345943214e-3,
        0.47008265848816866105e-5,
        0.24694040760197315333e-7,
        0.12418779768752299093e-9,
        0.59486890370320261949e-12,
        0.26957764568888888889e-14)
    IMPL_ERFCX_Y100_CASE(
        10,
        0.16540351739394069380e-1,
        0.87928458641241463952e-3,
        0.48520195793001753903e-5,
        0.25711774900881709176e-7,
        0.13030128534230822419e-9,
        0.62820097586874779402e-12,
        0.28612737351111111111e-14)
    IMPL_ERFCX_Y100_CASE(
        11,
        0.18318536789842392647e-1,
        0.89900542647891721692e-3,
        0.50094684089553365810e-5,
        0.26779777074218070482e-7,
        0.13675822186304615566e-9,
        0.66358287745352705725e-12,
        0.30375273884444444444e-14)
    IMPL_ERFCX_Y100_CASE(
        12,
        0.20136801964214276775e-1,
        0.91936908737673676012e-3,
        0.51734830914104276820e-5,
        0.27900878609710432673e-7,
        0.14357976402809042257e-9,
        0.70114790311043728387e-12,
        0.32252476000000000000e-14)
    IMPL_ERFCX_Y100_CASE(
        13,
        0.21996459598282740954e-1,
        0.94040248155366777784e-3,
        0.53443911508041164739e-5,
        0.29078085538049374673e-7,
        0.15078844500329731137e-9,
        0.74103813647499204269e-12,
        0.34251892320000000000e-14)
    IMPL_ERFCX_Y100_CASE(
        14,
        0.23898877187226319502e-1,
        0.96213386835900177540e-3,
        0.55225386998049012752e-5,
        0.30314589961047687059e-7,
        0.15840826497296335264e-9,
        0.78340500472414454395e-12,
        0.36381553564444444445e-14)
    IMPL_ERFCX_Y100_CASE(
        15,
        0.25845480155298518485e-1,
        0.98459293067820123389e-3,
        0.57082915920051843672e-5,
        0.31613782169164830118e-7,
        0.16646478745529630813e-9,
        0.82840985928785407942e-12,
        0.38649975768888888890e-14)
    IMPL_ERFCX_Y100_CASE(
        16,
        0.27837754783474696598e-1,
        0.10078108563256892757e-2,
        0.59020366493792212221e-5,
        0.32979263553246520417e-7,
        0.17498524159268458073e-9,
        0.87622459124842525110e-12,
        0.41066206488888888890e-14)
    IMPL_ERFCX_Y100_CASE(
        17,
        0.29877251304899307550e-1,
        0.10318204245057349310e-2,
        0.61041829697162055093e-5,
        0.34414860359542720579e-7,
        0.18399863072934089607e-9,
        0.92703227366365046533e-12,
        0.43639844053333333334e-14)
    IMPL_ERFCX_Y100_CASE(
        18,
        0.31965587178596443475e-1,
        0.10566560976716574401e-2,
        0.63151633192414586770e-5,
        0.35924638339521924242e-7,
        0.19353584758781174038e-9,
        0.98102783859889264382e-12,
        0.46381060817777777779e-14)
    IMPL_ERFCX_Y100_CASE(
        19,
        0.34104450552588334840e-1,
        0.10823541191350532574e-2,
        0.65354356159553934436e-5,
        0.37512918348533521149e-7,
        0.20362979635817883229e-9,
        0.10384187833037282363e-11,
        0.49300625262222222221e-14)
    IMPL_ERFCX_Y100_CASE(
        20,
        0.36295603928292425716e-1,
        0.11089526167995268200e-2,
        0.67654845095518363577e-5,
        0.39184292949913591646e-7,
        0.21431552202133775150e-9,
        0.10994259106646731797e-11,
        0.52409949102222222221e-14)
    IMPL_ERFCX_Y100_CASE(
        21,
        0.38540888038840509795e-1,
        0.11364917134175420009e-2,
        0.70058230641246312003e-5,
        0.40943644083718586939e-7,
        0.22563034723692881631e-9,
        0.11642841011361992885e-11,
        0.55721092871111111110e-14)
    IMPL_ERFCX_Y100_CASE(
        22,
        0.40842225954785960651e-1,
        0.11650136437945673891e-2,
        0.72569945502343006619e-5,
        0.42796161861855042273e-7,
        0.23761401711005024162e-9,
        0.12332431172381557035e-11,
        0.59246802364444444445e-14)
    IMPL_ERFCX_Y100_CASE(
        23,
        0.43201627431540222422e-1,
        0.11945628793917272199e-2,
        0.75195743532849206263e-5,
        0.44747364553960993492e-7,
        0.25030885216472953674e-9,
        0.13065684400300476484e-11,
        0.63000532853333333334e-14)
    IMPL_ERFCX_Y100_CASE(
        24,
        0.45621193513810471438e-1,
        0.12251862608067529503e-2,
        0.77941720055551920319e-5,
        0.46803119830954460212e-7,
        0.26375990983978426273e-9,
        0.13845421370977119765e-11,
        0.66996477404444444445e-14)
    IMPL_ERFCX_Y100_CASE(
        25,
        0.48103121413299865517e-1,
        0.12569331386432195113e-2,
        0.80814333496367673980e-5,
        0.48969667335682018324e-7,
        0.27801515481905748484e-9,
        0.14674637611609884208e-11,
        0.71249589351111111110e-14)
    IMPL_ERFCX_Y100_CASE(
        26,
        0.50649709676983338501e-1,
        0.12898555233099055810e-2,
        0.83820428414568799654e-5,
        0.51253642652551838659e-7,
        0.29312563849675507232e-9,
        0.15556512782814827846e-11,
        0.75775607822222222221e-14)
    IMPL_ERFCX_Y100_CASE(
        27,
        0.53263363664388864181e-1,
        0.13240082443256975769e-2,
        0.86967260015007658418e-5,
        0.53662102750396795566e-7,
        0.30914568786634796807e-9,
        0.16494420240828493176e-11,
        0.80591079644444444445e-14)
    IMPL_ERFCX_Y100_CASE(
        28,
        0.55946601353500013794e-1,
        0.13594491197408190706e-2,
        0.90262520233016380987e-5,
        0.56202552975056695376e-7,
        0.32613310410503135996e-9,
        0.17491936862246367398e-11,
        0.85713381688888888890e-14)
    IMPL_ERFCX_Y100_CASE(
        29,
        0.58702059496154081813e-1,
        0.13962391363223647892e-2,
        0.93714365487312784270e-5,
        0.58882975670265286526e-7,
        0.34414937110591753387e-9,
        0.18552853109751857859e-11,
        0.91160736711111111110e-14)
    IMPL_ERFCX_Y100_CASE(
        30,
        0.61532500145144778048e-1,
        0.14344426411912015247e-2,
        0.97331446201016809696e-5,
        0.61711860507347175097e-7,
        0.36325987418295300221e-9,
        0.19681183310134518232e-11,
        0.96952238400000000000e-14)
    IMPL_ERFCX_Y100_CASE(
        31,
        0.64440817576653297993e-1,
        0.14741275456383131151e-2,
        0.10112293819576437838e-4,
        0.64698236605933246196e-7,
        0.38353412915303665586e-9,
        0.20881176114385120186e-11,
        0.10310784480000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        32,
        0.67430045633130393282e-1,
        0.15153655418916540370e-2,
        0.10509857606888328667e-4,
        0.67851706529363332855e-7,
        0.40504602194811140006e-9,
        0.22157325110542534469e-11,
        0.10964842115555555556e-13)
    IMPL_ERFCX_Y100_CASE(
        33,
        0.70503365513338850709e-1,
        0.15582323336495709827e-2,
        0.10926868866865231089e-4,
        0.71182482239613507542e-7,
        0.42787405890153386710e-9,
        0.23514379522274416437e-11,
        0.11659571751111111111e-13)
    IMPL_ERFCX_Y100_CASE(
        34,
        0.73664114037944596353e-1,
        0.16028078812438820413e-2,
        0.11364423678778207991e-4,
        0.74701423097423182009e-7,
        0.45210162777476488324e-9,
        0.24957355004088569134e-11,
        0.12397238257777777778e-13)
    IMPL_ERFCX_Y100_CASE(
        35,
        0.76915792420819562379e-1,
        0.16491766623447889354e-2,
        0.11823685320041302169e-4,
        0.78420075993781544386e-7,
        0.47781726956916478925e-9,
        0.26491544403815724749e-11,
        0.13180196462222222222e-13)
    IMPL_ERFCX_Y100_CASE(
        36,
        0.80262075578094612819e-1,
        0.16974279491709504117e-2,
        0.12305888517309891674e-4,
        0.82350717698979042290e-7,
        0.50511496109857113929e-9,
        0.28122528497626897696e-11,
        0.14010889635555555556e-13)
    IMPL_ERFCX_Y100_CASE(
        37,
        0.83706822008980357446e-1,
        0.17476561032212656962e-2,
        0.12812343958540763368e-4,
        0.86506399515036435592e-7,
        0.53409440823869467453e-9,
        0.29856186620887555043e-11,
        0.14891851591111111111e-13)
    IMPL_ERFCX_Y100_CASE(
        38,
        0.87254084284461718231e-1,
        0.17999608886001962327e-2,
        0.13344443080089492218e-4,
        0.90900994316429008631e-7,
        0.56486134972616465316e-9,
        0.31698707080033956934e-11,
        0.15825697795555555556e-13)
    IMPL_ERFCX_Y100_CASE(
        39,
        0.90908120182172748487e-1,
        0.18544478050657699758e-2,
        0.13903663143426120077e-4,
        0.95549246062549906177e-7,
        0.59752787125242054315e-9,
        0.33656597366099099413e-11,
        0.16815130613333333333e-13)
    IMPL_ERFCX_Y100_CASE(
        40,
        0.94673404508075481121e-1,
        0.19112284419887303347e-2,
        0.14491572616545004930e-4,
        0.10046682186333613697e-6,
        0.63221272959791000515e-9,
        0.35736693975589130818e-11,
        0.17862931591111111111e-13)
    IMPL_ERFCX_Y100_CASE(
        41,
        0.98554641648004456555e-1,
        0.19704208544725622126e-2,
        0.15109836875625443935e-4,
        0.10567036667675984067e-6,
        0.66904168640019354565e-9,
        0.37946171850824333014e-11,
        0.18971959040000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        42,
        0.10255677889470089531e0,
        0.20321499629472857418e-2,
        0.15760224242962179564e-4,
        0.11117756071353507391e-6,
        0.70814785110097658502e-9,
        0.40292553276632563925e-11,
        0.20145143075555555556e-13)
    IMPL_ERFCX_Y100_CASE(
        43,
        0.10668502059865093318e0,
        0.20965479776148731610e-2,
        0.16444612377624983565e-4,
        0.11700717962026152749e-6,
        0.74967203250938418991e-9,
        0.42783716186085922176e-11,
        0.21385479360000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        44,
        0.11094484319386444474e0,
        0.21637548491908170841e-2,
        0.17164995035719657111e-4,
        0.12317915750735938089e-6,
        0.79376309831499633734e-9,
        0.45427901763106353914e-11,
        0.22696025653333333333e-13)
    IMPL_ERFCX_Y100_CASE(
        45,
        0.11534201115268804714e0,
        0.22339187474546420375e-2,
        0.17923489217504226813e-4,
        0.12971465288245997681e-6,
        0.84057834180389073587e-9,
        0.48233721206418027227e-11,
        0.24079890062222222222e-13)
    IMPL_ERFCX_Y100_CASE(
        46,
        0.11988259392684094740e0,
        0.23071965691918689601e-2,
        0.18722342718958935446e-4,
        0.13663611754337957520e-6,
        0.89028385488493287005e-9,
        0.51210161569225846701e-11,
        0.25540227111111111111e-13)
    IMPL_ERFCX_Y100_CASE(
        47,
        0.12457298393509812907e0,
        0.23837544771809575380e-2,
        0.19563942105711612475e-4,
        0.14396736847739470782e-6,
        0.94305490646459247016e-9,
        0.54366590583134218096e-11,
        0.27080225920000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        48,
        0.12941991566142438816e0,
        0.24637684719508859484e-2,
        0.20450821127475879816e-4,
        0.15173366280523906622e-6,
        0.99907632506389027739e-9,
        0.57712760311351625221e-11,
        0.28703099555555555556e-13)
    IMPL_ERFCX_Y100_CASE(
        49,
        0.13443048593088696613e0,
        0.25474249981080823877e-2,
        0.21385669591362915223e-4,
        0.15996177579900443030e-6,
        0.10585428844575134013e-8,
        0.61258809536787882989e-11,
        0.30412080142222222222e-13)
    IMPL_ERFCX_Y100_CASE(
        50,
        0.13961217543434561353e0,
        0.26349215871051761416e-2,
        0.22371342712572567744e-4,
        0.16868008199296822247e-6,
        0.11216596910444996246e-8,
        0.65015264753090890662e-11,
        0.32210394506666666666e-13)
    IMPL_ERFCX_Y100_CASE(
        51,
        0.14497287157673800690e0,
        0.27264675383982439814e-2,
        0.23410870961050950197e-4,
        0.17791863939526376477e-6,
        0.11886425714330958106e-8,
        0.68993039665054288034e-11,
        0.34101266222222222221e-13)
    IMPL_ERFCX_Y100_CASE(
        52,
        0.15052089272774618151e0,
        0.28222846410136238008e-2,
        0.24507470422713397006e-4,
        0.18770927679626136909e-6,
        0.12597184587583370712e-8,
        0.73203433049229821618e-11,
        0.36087889048888888890e-13)
    IMPL_ERFCX_Y100_CASE(
        53,
        0.15626501395774612325e0,
        0.29226079376196624949e-2,
        0.25664553693768450545e-4,
        0.19808568415654461964e-6,
        0.13351257759815557897e-8,
        0.77658124891046760667e-11,
        0.38173420035555555555e-13)
    IMPL_ERFCX_Y100_CASE(
        54,
        0.16221449434620737567e0,
        0.30276865332726475672e-2,
        0.26885741326534564336e-4,
        0.20908350604346384143e-6,
        0.14151148144240728728e-8,
        0.82369170665974313027e-11,
        0.40360957457777777779e-13)
    IMPL_ERFCX_Y100_CASE(
        55,
        0.16837910595412130659e0,
        0.31377844510793082301e-2,
        0.28174873844911175026e-4,
        0.22074043807045782387e-6,
        0.14999481055996090039e-8,
        0.87348993661930809254e-11,
        0.42653528977777777779e-13)
    IMPL_ERFCX_Y100_CASE(
        56,
        0.17476916455659369953e0,
        0.32531815370903068316e-2,
        0.29536024347344364074e-4,
        0.23309632627767074202e-6,
        0.15899007843582444846e-8,
        0.92610375235427359475e-11,
        0.45054073102222222221e-13)
    IMPL_ERFCX_Y100_CASE(
        57,
        0.18139556223643701364e0,
        0.33741744168096996041e-2,
        0.30973511714709500836e-4,
        0.24619326937592290996e-6,
        0.16852609412267750744e-8,
        0.98166442942854895573e-11,
        0.47565418097777777779e-13)
    IMPL_ERFCX_Y100_CASE(
        58,
        0.18826980194443664549e0,
        0.35010775057740317997e-2,
        0.32491914440014267480e-4,
        0.26007572375886319028e-6,
        0.17863299617388376116e-8,
        0.10403065638343878679e-10,
        0.50190265831111111110e-13)
    IMPL_ERFCX_Y100_CASE(
        59,
        0.19540403413693967350e0,
        0.36342240767211326315e-2,
        0.34096085096200907289e-4,
        0.27479061117017637474e-6,
        0.18934228504790032826e-8,
        0.11021679075323598664e-10,
        0.52931171733333333334e-13)
    IMPL_ERFCX_Y100_CASE(
        60,
        0.20281109560651886959e0,
        0.37739673859323597060e-2,
        0.35791165457592409054e-4,
        0.29038742889416172404e-6,
        0.20068685374849001770e-8,
        0.11673891799578381999e-10,
        0.55790523093333333334e-13)
    IMPL_ERFCX_Y100_CASE(
        61,
        0.21050455062669334978e0,
        0.39206818613925652425e-2,
        0.37582602289680101704e-4,
        0.30691836231886877385e-6,
        0.21270101645763677824e-8,
        0.12361138551062899455e-10,
        0.58770520160000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        62,
        0.21849873453703332479e0,
        0.40747643554689586041e-2,
        0.39476163820986711501e-4,
        0.32443839970139918836e-6,
        0.22542053491518680200e-8,
        0.13084879235290858490e-10,
        0.61873153262222222221e-13)
    IMPL_ERFCX_Y100_CASE(
        63,
        0.22680879990043229327e0,
        0.42366354648628516935e-2,
        0.41477956909656896779e-4,
        0.34300544894502810002e-6,
        0.23888264229264067658e-8,
        0.13846596292818514601e-10,
        0.65100183751111111110e-13)
    IMPL_ERFCX_Y100_CASE(
        64,
        0.23545076536988703937e0,
        0.44067409206365170888e-2,
        0.43594444916224700881e-4,
        0.36268045617760415178e-6,
        0.25312606430853202748e-8,
        0.14647791812837903061e-10,
        0.68453122631111111110e-13)
    IMPL_ERFCX_Y100_CASE(
        65,
        0.24444156740777432838e0,
        0.45855530511605787178e-2,
        0.45832466292683085475e-4,
        0.38352752590033030472e-6,
        0.26819103733055603460e-8,
        0.15489984390884756993e-10,
        0.71933206364444444445e-13)
    IMPL_ERFCX_Y100_CASE(
        66,
        0.25379911500634264643e0,
        0.47735723208650032167e-2,
        0.48199253896534185372e-4,
        0.40561404245564732314e-6,
        0.28411932320871165585e-8,
        0.16374705736458320149e-10,
        0.75541379822222222221e-13)
    IMPL_ERFCX_Y100_CASE(
        67,
        0.26354234756393613032e0,
        0.49713289477083781266e-2,
        0.50702455036930367504e-4,
        0.42901079254268185722e-6,
        0.30095422058900481753e-8,
        0.17303497025347342498e-10,
        0.79278273368888888890e-13)
    IMPL_ERFCX_Y100_CASE(
        68,
        0.27369129607732343398e0,
        0.51793846023052643767e-2,
        0.53350152258326602629e-4,
        0.45379208848865015485e-6,
        0.31874057245814381257e-8,
        0.18277905010245111046e-10,
        0.83144182364444444445e-13)
    IMPL_ERFCX_Y100_CASE(
        69,
        0.28426714781640316172e0,
        0.53983341916695141966e-2,
        0.56150884865255810638e-4,
        0.48003589196494734238e-6,
        0.33752476967570796349e-8,
        0.19299477888083469086e-10,
        0.87139049137777777779e-13)
    IMPL_ERFCX_Y100_CASE(
        70,
        0.29529231465348519920e0,
        0.56288077305420795663e-2,
        0.59113671189913307427e-4,
        0.50782393781744840482e-6,
        0.35735475025851713168e-8,
        0.20369760937017070382e-10,
        0.91262442613333333334e-13)
    IMPL_ERFCX_Y100_CASE(
        71,
        0.30679050522528838613e0,
        0.58714723032745403331e-2,
        0.62248031602197686791e-4,
        0.53724185766200945789e-6,
        0.37827999418960232678e-8,
        0.21490291930444538307e-10,
        0.95513539182222222221e-13)
    IMPL_ERFCX_Y100_CASE(
        72,
        0.31878680111173319425e0,
        0.61270341192339103514e-2,
        0.65564012259707640976e-4,
        0.56837930287837738996e-6,
        0.40035151353392378882e-8,
        0.22662596341239294792e-10,
        0.99891109760000000000e-13)
    IMPL_ERFCX_Y100_CASE(
        73,
        0.33130773722152622027e0,
        0.63962406646798080903e-2,
        0.69072209592942396666e-4,
        0.60133006661885941812e-6,
        0.42362183765883466691e-8,
        0.23888182347073698382e-10,
        0.10439349811555555556e-12)
    IMPL_ERFCX_Y100_CASE(
        74,
        0.34438138658041336523e0,
        0.66798829540414007258e-2,
        0.72783795518603561144e-4,
        0.63619220443228800680e-6,
        0.44814499336514453364e-8,
        0.25168535651285475274e-10,
        0.10901861383111111111e-12)
    IMPL_ERFCX_Y100_CASE(
        75,
        0.35803744972380175583e0,
        0.69787978834882685031e-2,
        0.76710543371454822497e-4,
        0.67306815308917386747e-6,
        0.47397647975845228205e-8,
        0.26505114141143050509e-10,
        0.11376390933333333333e-12)
    IMPL_ERFCX_Y100_CASE(
        76,
        0.37230734890119724188e0,
        0.72938706896461381003e-2,
        0.80864854542670714092e-4,
        0.71206484718062688779e-6,
        0.50117323769745883805e-8,
        0.27899342394100074165e-10,
        0.11862637614222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        77,
        0.38722432730555448223e0,
        0.76260375162549802745e-2,
        0.85259785810004603848e-4,
        0.75329383305171327677e-6,
        0.52979361368388119355e-8,
        0.29352606054164086709e-10,
        0.12360253370666666667e-12)
    IMPL_ERFCX_Y100_CASE(
        78,
        0.40282355354616940667e0,
        0.79762880915029728079e-2,
        0.89909077342438246452e-4,
        0.79687137961956194579e-6,
        0.55989731807360403195e-8,
        0.30866246101464869050e-10,
        0.12868841946666666667e-12)
    IMPL_ERFCX_Y100_CASE(
        79,
        0.41914223158913787649e0,
        0.83456685186950463538e-2,
        0.94827181359250161335e-4,
        0.84291858561783141014e-6,
        0.59154537751083485684e-8,
        0.32441553034347469291e-10,
        0.13387957943111111111e-12)
    IMPL_ERFCX_Y100_CASE(
        80,
        0.43621971639463786896e0,
        0.87352841828289495773e-2,
        0.10002929142066799966e-3,
        0.89156148280219880024e-6,
        0.62480008150788597147e-8,
        0.34079760983458878910e-10,
        0.13917107176888888889e-12)
    IMPL_ERFCX_Y100_CASE(
        81,
        0.45409763548534330981e0,
        0.91463027755548240654e-2,
        0.10553137232446167258e-3,
        0.94293113464638623798e-6,
        0.65972492312219959885e-8,
        0.35782041795476563662e-10,
        0.14455745872000000000e-12)
    IMPL_ERFCX_Y100_CASE(
        82,
        0.47282001668512331468e0,
        0.95799574408860463394e-2,
        0.11135019058000067469e-3,
        0.99716373005509038080e-6,
        0.69638453369956970347e-8,
        0.37549499088161345850e-10,
        0.15003280712888888889e-12)
    IMPL_ERFCX_Y100_CASE(
        83,
        0.49243342227179841649e0,
        0.10037550043909497071e-1,
        0.11750334542845234952e-3,
        0.10544006716188967172e-5,
        0.73484461168242224872e-8,
        0.39383162326435752965e-10,
        0.15559069118222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        84,
        0.51298708979209258326e0,
        0.10520454564612427224e-1,
        0.12400930037494996655e-3,
        0.11147886579371265246e-5,
        0.77517184550568711454e-8,
        0.41283980931872622611e-10,
        0.16122419680000000000e-12)
    IMPL_ERFCX_Y100_CASE(
        85,
        0.53453307979101369843e0,
        0.11030120618800726938e-1,
        0.13088741519572269581e-3,
        0.11784797595374515432e-5,
        0.81743383063044825400e-8,
        0.43252818449517081051e-10,
        0.16692592640000000000e-12)
    IMPL_ERFCX_Y100_CASE(
        86,
        0.55712643071169299478e0,
        0.11568077107929735233e-1,
        0.13815797838036651289e-3,
        0.12456314879260904558e-5,
        0.86169898078969313597e-8,
        0.45290446811539652525e-10,
        0.17268801084444444444e-12)
    IMPL_ERFCX_Y100_CASE(
        87,
        0.58082532122519320968e0,
        0.12135935999503877077e-1,
        0.14584223996665838559e-3,
        0.13164068573095710742e-5,
        0.90803643355106020163e-8,
        0.47397540713124619155e-10,
        0.17850211608888888889e-12)
    IMPL_ERFCX_Y100_CASE(
        88,
        0.60569124025293375554e0,
        0.12735396239525550361e-1,
        0.15396244472258863344e-3,
        0.13909744385382818253e-5,
        0.95651595032306228245e-8,
        0.49574672127669041550e-10,
        0.18435945564444444444e-12)
    IMPL_ERFCX_Y100_CASE(
        89,
        0.63178916494715716894e0,
        0.13368247798287030927e-1,
        0.16254186562762076141e-3,
        0.14695084048334056083e-5,
        0.10072078109604152350e-7,
        0.51822304995680707483e-10,
        0.19025081422222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        90,
        0.65918774689725319200e0,
        0.14036375850601992063e-1,
        0.17160483760259706354e-3,
        0.15521885688723188371e-5,
        0.10601827031535280590e-7,
        0.54140790105837520499e-10,
        0.19616655146666666667e-12)
    IMPL_ERFCX_Y100_CASE(
        91,
        0.68795950683174433822e0,
        0.14741765091365869084e-1,
        0.18117679143520433835e-3,
        0.16392004108230585213e-5,
        0.11155116068018043001e-7,
        0.56530360194925690374e-10,
        0.20209663662222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        92,
        0.71818103808729967036e0,
        0.15486504187117112279e-1,
        0.19128428784550923217e-3,
        0.17307350969359975848e-5,
        0.11732656736113607751e-7,
        0.58991125287563833603e-10,
        0.20803065333333333333e-12)
    IMPL_ERFCX_Y100_CASE(
        93,
        0.74993321911726254661e0,
        0.16272790364044783382e-1,
        0.20195505163377912645e-3,
        0.18269894883203346953e-5,
        0.12335161021630225535e-7,
        0.61523068312169087227e-10,
        0.21395783431111111111e-12)
    IMPL_ERFCX_Y100_CASE(
        94,
        0.78330143531283492729e0,
        0.17102934132652429240e-1,
        0.21321800585063327041e-3,
        0.19281661395543913713e-5,
        0.12963340087354341574e-7,
        0.64126040998066348872e-10,
        0.21986708942222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        95,
        0.81837581041023811832e0,
        0.17979364149044223802e-1,
        0.22510330592753129006e-3,
        0.20344732868018175389e-5,
        0.13617902941839949718e-7,
        0.66799760083972474642e-10,
        0.22574701262222222222e-12)
    IMPL_ERFCX_Y100_CASE(
        96,
        0.85525144775685126237e0,
        0.18904632212547561026e-1,
        0.23764237370371255638e-3,
        0.21461248251306387979e-5,
        0.14299555071870523786e-7,
        0.69543803864694171934e-10,
        0.23158593688888888889e-12)
    IMPL_ERFCX_Y100_CASE(
        97,
        0.89402868170849933734e0,
        0.19881418399127202569e-1,
        0.25086793128395995798e-3,
        0.22633402747585233180e-5,
        0.15008997042116532283e-7,
        0.72357609075043941261e-10,
        0.23737194737777777778e-12)
    IMPL_ERFCX_Y100_CASE(
        98,
        0.93481333942870796363e0,
        0.20912536329780368893e-1,
        0.26481403465998477969e-3,
        0.23863447359754921676e-5,
        0.15746923065472184451e-7,
        0.75240468141720143653e-10,
        0.24309291271111111111e-12)
    IMPL_ERFCX_Y100_CASE(
        99,
        0.97771701335885035464e0,
        0.22000938572830479551e-1,
        0.27951610702682383001e-3,
        0.25153688325245314530e-5,
        0.16514019547822821453e-7,
        0.78191526829368231251e-10,
        0.24873652355555555556e-12)
  }
  // we only get here if y = 1, i.e. |x| < 4*eps, in which case
  // erfcx is within 1e-15 of 1..
  return 1.0;
}

template <typename T>
float erfcx(T x) {
  if (x != x) {
    return x;
  }

  if (x >= 0) {
    if (x > 50) { // continued-fraction expansion is faster
      const auto ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
      if (x > 5e7) { // 1-term expansion, important to avoid overflow
        return ispi / x;
      }
      /* 5-term expansion (rely on compiler for CSE), simplified from:
                ispi / (x+0.5/(x+1/(x+1.5/(x+2/x))))  */
      return ispi * ((x * x) * (x * x + 4.5) + 2) /
          (x * ((x * x) * (x * x + 5) + 3.75));
    }
    return erfcx_y100(400.0f / (4.0f + x));
  } else {
    if (x < -26.7) {
      return ::metal::numeric_limits<float>::infinity();
    } else if (x < -6.1) {
      return 2 * exp(float(x) * x);
    } else {
      return 2 * exp(float(x) * x) - erfcx_y100(400.0f / (4 - x));
    }
  }
}

} // namespace metal
} // namespace c10
