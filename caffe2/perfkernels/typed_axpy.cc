#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/core/types.h"
#include "caffe2/perfkernels/common.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
void TypedAxpy<float, float>(int N, const float a, const float* x, float* y) {
  // This uses a hack that axpy implementation actually does not use the
  // CPUContext, so passing in a nullpointer works.
  math::Axpy<float, CPUContext>(N, a, x, y, nullptr);
}

void TypedAxpy_float16_float__base(
    int N,
    const float a,
    const float16* x,
    float* y) {
  for (int i = 0; i < N; ++i) {
    union {
      uint32_t intval;
      float floatval;
    } t1;
    uint32_t t2, t3;
    t1.intval = x[i].x & 0x7fff; // Non-sign bits
    t2 = x[i].x & 0x8000; // Sign bit
    t3 = x[i].x & 0x7c00; // Exponent
    t1.intval <<= 13; // Align mantissa on MSB
    t2 <<= 16; // Shift sign bit into position
    t1.intval += 0x38000000; // Adjust bias
    t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
    t1.intval |= t2; // Re-insert sign bit
    y[i] += t1.floatval * a;
  }
}

template <>
void TypedAxpy<float16, float>(
    int N,
    const float a,
    const float16* x,
    float* y) {
  AVX2_FMA_DO(TypedAxpy_float16_float, N, a, x, y);
  AVX_F16C_DO(TypedAxpy_float16_float, N, a, x, y);
  BASE_DO(TypedAxpy_float16_float, N, a, x, y);
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
