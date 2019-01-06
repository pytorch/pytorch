#pragma once

#include <ATen/core/Half.h>

#include "caffe2/perfkernels/typed_axpy.h"

namespace caffe2 {

void TypedAxpyHalffloat__avx2_fma(
    int N,
    const float a,
    const at::Half* x,
    float* y);

void TypedAxpy_uint8_float__avx2_fma(
    int N,
    const float a,
    const std::uint8_t* x,
    float* y);

} // namespace caffe2
