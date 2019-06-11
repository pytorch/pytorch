#pragma once

#include <cstdint>

namespace caffe2 {

/**
 * Optimized using AVX2 intrinsics for max pool 2D in NHWC layout
 */
void max_pool_avx2(
    const std::uint8_t* Xdata,
    int n,
    int height,
    int width,
    int channels,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    std::uint8_t* Ydata);

} // namespace caffe2
