#include <c10/util/Half.h>
#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/perfkernels/common.h"

namespace caffe2 {

void TypedAxpy__base(int N, const float a, const float* x, float* y) {
  for (int i = 0; i < N; ++i) {
    y[i] += a * x[i];
  }
}

decltype(TypedAxpy__base) TypedAxpy__avx2_fma;
decltype(TypedAxpy__base) TypedAxpy__avx_f16c;
template <>
void TypedAxpy<float, float>(int N, const float a, const float* x, float* y) {
  AVX2_FMA_DO(TypedAxpy, N, a, x, y);
  AVX_F16C_DO(TypedAxpy, N, a, x, y);
  BASE_DO(TypedAxpy, N, a, x, y);
}

void TypedAxpyHalffloat__base(
    int N,
    const float a,
    const at::Half* x,
    float* y) {
  for (int i = 0; i < N; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t intval;
      float floatval;
    } t1;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t t2, t3;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t1.intval = x[i].x & 0x7fff; // Non-sign bits
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t2 = x[i].x & 0x8000; // Sign bit
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t3 = x[i].x & 0x7c00; // Exponent
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t1.intval <<= 13; // Align mantissa on MSB
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t2 <<= 16; // Shift sign bit into position
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    t1.intval += 0x38000000; // Adjust bias
    t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
    t1.intval |= t2; // Re-insert sign bit
    y[i] += t1.floatval * a;
  }
}

decltype(TypedAxpyHalffloat__base) TypedAxpyHalffloat__avx2_fma;
decltype(TypedAxpyHalffloat__base) TypedAxpyHalffloat__avx_f16c;
template <>
void TypedAxpy<at::Half, float>(
    int N,
    const float a,
    const at::Half* x,
    float* y) {
  AVX2_FMA_DO(TypedAxpyHalffloat, N, a, x, y);
  AVX_F16C_DO(TypedAxpyHalffloat, N, a, x, y);
  BASE_DO(TypedAxpyHalffloat, N, a, x, y);
}

void TypedAxpy_uint8_float__base(
    int N,
    const float a,
    const std::uint8_t* x,
    float* y) {
  for (int i = 0; i < N; ++i) {
    y[i] += (float)(x[i]) * a;
  }
}

decltype(TypedAxpy_uint8_float__base) TypedAxpy_uint8_float__avx2_fma;
decltype(TypedAxpy_uint8_float__base) TypedAxpy_uint8_float__avx_f16c;
template <>
void TypedAxpy<std::uint8_t, float>(
    int N,
    const float a,
    const std::uint8_t* x,
    float* y) {
  AVX2_FMA_DO(TypedAxpy_uint8_float, N, a, x, y);
  BASE_DO(TypedAxpy_uint8_float, N, a, x, y);
}

} // namespace caffe2
