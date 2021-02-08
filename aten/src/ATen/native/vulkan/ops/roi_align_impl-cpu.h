// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef ROI_ALIGN_IMPL_CPU_H
#define ROI_ALIGN_IMPL_CPU_H

#include <assert.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace at {
namespace native {

template <typename T>
struct BilinearInterpolationParam {
  int64_t p1;
  int64_t p2;
  int64_t p3;
  int64_t p4;
  T w1;
  T w2;
  T w3;
  T w4;
};

void ROIAlignForwardCpuImplWithNCHW(
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t roi_cols,
    int64_t pooled_h,
    int64_t pooled_w,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    const float* X,
    const float* R,
    float* Y);

} // namespace native
} // namespace at

#endif
