#pragma once

#include <ATen/AccumulateType.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

/*
* The following function was converted to CUDA form from code that comes
* with the following copyright notice. It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
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
* The following function was converted to CUDA form from code that comes
* with the following copyright notice. It has been released under the BSD license.
*
* Cephes Math Library Release 2.8:  June, 2000
* Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
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

}
}
