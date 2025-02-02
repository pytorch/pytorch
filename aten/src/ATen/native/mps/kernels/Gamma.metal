/*
 * The gamma function approximations follow John D Cook's
 * c++ implementation:  https://www.johndcook.com/Gamma.cpp.
 * (BSD License)
 *
 *
 * The digamma kernel and helper function is derived from the pytorch cpu
 * of this function, which is itself derived from the implementation
 * of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */

#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;

float calc_digamma_positive_domain(float x) {
  const float DIGAMMA_COEF[7] = {
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
      y += pow(z, i) * DIGAMMA_COEF[i];
    }
    y *= z;
  }
  return result + log(x) - (0.5 / x) - y;
}

float calc_trigamma(float x) {
  float sign = 1.0f;
  float result = 0.0f;

  if (x < 0.0f) {
    sign = -1.0f;
    auto sin_pi_x = sin(M_PI_F * x);
    result -= (M_PI_F * M_PI_F) / (sin_pi_x * sin_pi_x);
    x = 1.0f - x;
  }

  else if (x == 0.0) {
    return INFINITY;
  }

  else if (x < 1.0) {
    result += 1.0 / (x * x);
    x += 1.0f;
  }

  for (int i = 0; i < 6; ++i) {
    result += 1.0f / (x * x);
    x += 1.0f;
  }

  const float ixx = 1.0f / (x * x);
  result +=
      (1.0f + 1.0f / (2.0f * x) +
       ixx * ((1.0f / 6.0f) - ixx * ((1.0f / 30.0f) - ixx * (1.0f / 42.0f)))) /
      x;
  return sign * result;
}

template <typename T0, typename T1>
kernel void lgamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  output[id] =
      static_cast<T1>(c10::metal::log_gamma(static_cast<float>(input[id])));
}

template <typename T0, typename T1>
kernel void digamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  float x = input[id];
  if (x < 0.0f) {
    if (x == trunc(x)) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      output[id] = static_cast<T1>(NAN);
    } else {
      // Extracts the fractional part of x as r, since tan(pi * r) is more
      // numerically accurate than tan(pi * x). While these operations are
      // mathematically equivalent since both x and r are in radians and tan()
      // has a periodicity of pi, in practice the computation of pi * x is a
      // source of error (when |x| > 1).
      float r = fract(x);
      output[id] = static_cast<T1>(
          calc_digamma_positive_domain(1.0f - x) - M_PI_F / tan(M_PI_F * r));
    }
  } else if (x == 0.0f) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    output[id] = static_cast<T1>(copysign(INFINITY, -x));
  } else {
    output[id] = static_cast<T1>(calc_digamma_positive_domain(x));
  }
}

template <typename T0, typename T1>
kernel void trigamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  float x = input[id];
  output[id] = static_cast<T1>(calc_trigamma(x));
}

template <typename T0, typename T1>
kernel void polygamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    constant int64_t& order [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  // already blocked if n <= 1
  output[id] = static_cast<T1>(c10::metal::polygamma(input[id], order));
}

#define INSTANTIATE_GAMMA_KERNELS(DTYPE0, DTYPE1)                             \
  template [[host_name("lgamma_" #DTYPE0 "_" #DTYPE1)]] kernel void lgamma(   \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("digamma_" #DTYPE0 "_" #DTYPE1)]] kernel void digamma( \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("trigamma_" #DTYPE0 "_" #DTYPE1)]] kernel void         \
  trigamma(                                                                   \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("polygamma_" #DTYPE0 "_" #DTYPE1)]] kernel void        \
  polygamma(                                                                  \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      constant int64_t& order [[buffer(2)]],                                  \
      uint id [[thread_position_in_grid]]);

#if __METAL_VERSION__ >= 310
INSTANTIATE_GAMMA_KERNELS(bfloat, bfloat);
#endif
INSTANTIATE_GAMMA_KERNELS(half, half);
INSTANTIATE_GAMMA_KERNELS(float, float);
INSTANTIATE_GAMMA_KERNELS(bool, float);
INSTANTIATE_GAMMA_KERNELS(uchar, float);
INSTANTIATE_GAMMA_KERNELS(char, float);
INSTANTIATE_GAMMA_KERNELS(short, float);
INSTANTIATE_GAMMA_KERNELS(int, float);
INSTANTIATE_GAMMA_KERNELS(long, float);
