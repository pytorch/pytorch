#pragma once
#ifdef __AVX__
#include <immintrin.h>
#endif
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace internal {

// Z=X*Y
template <typename XT, typename YT, typename ZT>
void dot(const int N, const XT* x, const YT* y, ZT* z, CPUContext* ctx) {
  CAFFE_THROW("Unsupported, see specialized implementations");
}

template <>
void dot<float, float, float>(
    const int N,
    const float* x,
    const float* y,
    float* z,
    CPUContext* ctx);

template <>
void dot<float, at::Half, float>(
    const int N,
    const float* x,
    const at::Half* y,
    float* z,
    CPUContext* ctx);

} // namespace internal
} // namespace caffe2
