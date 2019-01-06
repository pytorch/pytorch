#pragma once

#include <ATen/core/Half.h>

#include "caffe2/perfkernels/typed_axpy.h"

namespace caffe2 {

void TypedAxpyHalffloat__avx_f16c(
    int N,
    const float a,
    const at::Half* x,
    float* y);

} // namespace caffe2
