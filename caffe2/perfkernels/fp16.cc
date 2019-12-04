#include "caffe2/perfkernels/fp16.h"
#include "caffe2/perfkernels/common.h"
#include "fbgemm/FbgemmConvert.h"

namespace caffe2 {

namespace {
void FloatToFloat16__base(const float* src, at::Half* dst, int size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

void FloatToFloat16__avx2_fma(const float* src, at::Half* dst, int size) {
  fbgemm::FloatToFloat16_simd(
      src, reinterpret_cast<fbgemm::float16*>(dst), size);
}

void Float16ToFloat__base(const at::Half* src, float* dst, int size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

void Float16ToFloat__avx2_fma(const at::Half* src, float* dst, int size) {
  fbgemm::Float16ToFloat_simd(
      reinterpret_cast<const fbgemm::float16*>(src), dst, size);
}
} // anonymous namespace

void FloatToFloat16(const float* src, at::Half* dst, int size) {
  AVX2_FMA_DO(FloatToFloat16, src, dst, size);
  BASE_DO(FloatToFloat16, src, dst, size);
}

void Float16ToFloat(const at::Half* src, float* dst, int size) {
  AVX2_FMA_DO(Float16ToFloat, src, dst, size);
  BASE_DO(Float16ToFloat, src, dst, size);
}

} // namespace caffe2
