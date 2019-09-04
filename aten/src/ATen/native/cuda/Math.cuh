#pragma once
#include <type_traits>

namespace at {
namespace native {

/*
* The following function was converted to CUDA form from code that comes
* with the following copyright notice. It has been released under the BSD license.
*
* Cephes Math Library Release 2.8:  June, 2000
* Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
*/
template <typename T, typename accreal>
__device__ __forceinline__ T digamma(T* in) {
  using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
  static const double PI_f64 = 3.14159265358979323846;
  static const compute_type PSI_10 = 2.25175258906672110764;
  static const compute_type A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  auto x = scalar_cast<compute_type>(*in);
  if (x == 0) {
    return scalar_cast<T>(INFINITY);
  }

  bool x_is_integer = x == floor(x);
  compute_type result = 0;
  if (x < 0) {
    if (x_is_integer) {
      return scalar_cast<T>(INFINITY);
    }
    // Rounding errors in tan's input can really affect the output
    // for extreme values, so we always perform this computation in double.
    result = scalar_cast<compute_type>(
        - PI_f64 / tan(PI_f64 * scalar_cast<double>(x)));
    x = 1 - x;
  }

  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return scalar_cast<T>(result + PSI_10);
  }

  compute_type y = 0;
  if (x < 1.0e17) {
    compute_type z = 1.0 / (x * x);

    compute_type polevl_result = 0;
    for (int i = 0; i <= 6; i++) {
      polevl_result = polevl_result * z + A[i];
    }
    y = z * polevl_result;
  }

  return scalar_cast<T>(log(x) - (0.5 / x) - y + result);
}

template <typename T, typename accreal>
__device__ __forceinline__ T trigamma(T* in) {
  using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
  const compute_type PI = 3.14159265358979323846;
  compute_type x = ScalarConvert<T, compute_type>::to(*in);
  compute_type sign = +1;
  compute_type result = 0;
  if (x < 0.5f) {
    sign = -1;
    compute_type sin_pi_x = THCNumerics<compute_type>::sin(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const compute_type ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1.f/6 - ixx * (1.f/30 - ixx * (1.f/42)))) / x;
  return ScalarConvert<compute_type, T>::to(sign * result);
}

}
}
