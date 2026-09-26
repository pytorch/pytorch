#pragma once
// calc_ndtri, shared verbatim between the CPU and the Metal shader compiler.
// ATen/native/Math.h wraps it for the host and CUDA, c10/metal/special_math.h
// exposes it as c10::metal::ndtri, which is the name the MPS backend of
// torch/_inductor/codegen/mps.py generates calls to. It lives in c10::detail
// because namespace at pulls in every name from c10, so a c10::calc_ndtri
// would make the unqualified calls in at::native ambiguous.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define C10_NDTRI_HOST_DEVICE
#define C10_NDTRI_STATIC
#define C10_NDTRI_THREAD thread
#define C10_NDTRI_LOG ::metal::precise::log
#define C10_NDTRI_SQRT ::metal::precise::sqrt
#define C10_NDTRI_INF(T) INFINITY
#define C10_NDTRI_NAN(T) NAN
#else
#include <c10/macros/Macros.h>
#include <cmath>
#include <cstddef>
#include <limits>
#define C10_NDTRI_HOST_DEVICE C10_HOST_DEVICE
#define C10_NDTRI_STATIC static
#define C10_NDTRI_THREAD
#define C10_NDTRI_LOG ::log
#define C10_NDTRI_SQRT ::sqrt
#define C10_NDTRI_INF(T) std::numeric_limits<T>::infinity()
#define C10_NDTRI_NAN(T) std::numeric_limits<T>::quiet_NaN()
#endif

// Metal 3 is C++14, so the namespaces cannot be concatenated here.
// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace c10 {
namespace detail {

// The Cephes coefficient tables stay C arrays: std::array does not exist in
// the Metal shading language, which compiles this header too.
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

/*
 * This function is derived from the implementation of the digamma function in
 * the Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 */
template <typename T>
C10_NDTRI_HOST_DEVICE inline T polevl(
    const T x,
    const C10_NDTRI_THREAD T A[],
    size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

/*
 * This function is derived from the implementation of the i1e function in the
 * Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Computes the argument, x, for which the area under the Gaussian probability
 * density function (integrated from minus infinity to x) is equal to y.
 */
template <typename T>
C10_NDTRI_HOST_DEVICE inline T calc_ndtri(T y0) {
  /* sqrt(2pi) */
  constexpr T s2pi = 2.50662827463100050242E0;
  constexpr T one = 1;
  constexpr T zero = 0;

  /* approximation for 0 <= |y - 0.5| <= 3/8 */
  C10_NDTRI_STATIC const T P0[5] = {
      -5.99633501014107895267E1,
      9.80010754185999661536E1,
      -5.66762857469070293439E1,
      1.39312609387279679503E1,
      -1.23916583867381258016E0,
  };

  C10_NDTRI_STATIC const T Q0[9] = {
      1.00000000000000000000E0,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
   * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
   */
  C10_NDTRI_STATIC const T P1[9] = {
      4.05544892305962419923E0,
      3.15251094599893866154E1,
      5.71628192246421288162E1,
      4.40805073893200834700E1,
      1.46849561928858024014E1,
      2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
  };

  C10_NDTRI_STATIC const T Q1[9] = {
      1.00000000000000000000E0,
      1.57799883256466749731E1,
      4.53907635128879210584E1,
      4.13172038254672030440E1,
      1.50425385692907503408E1,
      2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
   * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
   */

  C10_NDTRI_STATIC const T P2[9] = {
      3.23774891776946035970E0,
      6.91522889068984211695E0,
      3.93881025292474443415E0,
      1.33303460815807542389E0,
      2.01485389549179081538E-1,
      1.23716634817820021358E-2,
      3.01581553508235416007E-4,
      2.65806974686737550832E-6,
      6.23974539184983293730E-9,
  };

  C10_NDTRI_STATIC const T Q2[9] = {
      1.00000000000000000000E0,
      6.02427039364742014255E0,
      3.67983563856160859403E0,
      1.37702099489081330271E0,
      2.16236993594496635890E-1,
      1.34204006088543189037E-2,
      3.28014464682127739104E-4,
      2.89247864745380683936E-6,
      6.79019408009981274425E-9,
  };

  if (y0 == zero) {
    return -C10_NDTRI_INF(T);
  }
  if (y0 == one) {
    return C10_NDTRI_INF(T);
  }
  if (y0 < zero || y0 > one) {
    return C10_NDTRI_NAN(T);
  }
  bool code = true;
  T y = y0;
  if (y > one - T{0.13533528323661269189}) { /* 0.135... = exp(-2) */
    y = one - y;
    code = false;
  }

  if (y > T{0.13533528323661269189}) {
    y = y - T{0.5};
    const T y2 = y * y;
    T x = y + y * (y2 * polevl(y2, P0, 4) / polevl(y2, Q0, 8));
    return (x * s2pi);
  }

  T x = C10_NDTRI_SQRT(T{-2.0} * C10_NDTRI_LOG(y));
  const T x0 = x - C10_NDTRI_LOG(x) / x;

  const T z = one / x;
  T x1;
  if (x < T{8.0}) /* y > exp(-32) = 1.2664165549e-14 */
  {
    x1 = z * polevl(z, P1, 8) / polevl(z, Q1, 8);
  } else {
    x1 = z * polevl(z, P2, 8) / polevl(z, Q2, 8);
  }
  x = x0 - x1;
  if (code) {
    x = -x;
  }
  return x;
}

// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

} // namespace detail
} // namespace c10

#undef C10_NDTRI_HOST_DEVICE
#undef C10_NDTRI_STATIC
#undef C10_NDTRI_THREAD
#undef C10_NDTRI_LOG
#undef C10_NDTRI_SQRT
#undef C10_NDTRI_INF
#undef C10_NDTRI_NAN
