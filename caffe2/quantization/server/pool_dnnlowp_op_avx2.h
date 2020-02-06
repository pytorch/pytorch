#pragma once

#include <cstdint>
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

/**
 * Optimized using AVX2 intrinsics for max pool 2D in NHWC layout
 */
CAFFE2_API void max_pool_avx2(
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

CAFFE2_API void average_pool_avx2(
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
    std::uint8_t* Ydata,
    float in_scale,
    float out_scale,
    int32_t in_zero_point,
    int32_t out_zero_point,
    int32_t minimum,
    int32_t maximum);

CAFFE2_API void average_pool_3d_avx2(
    const uint8_t* Xdata,
    int n,
    int height,
    int width,
    int depth,
    int channels,
    int pooled_height,
    int pooled_width,
    int pooled_depth,
    int kernel_h,
    int kernel_w,
    int kernel_d,
    int stride_h,
    int stride_w,
    int stride_d,
    int pad_t,
    int pad_l,
    int pad_d,
    uint8_t* Ydata,
    float in_scale,
    float out_scale,
    int32_t in_zero_point,
    int32_t out_zero_point,
    int32_t minimum,
    int32_t maximum);

} // namespace caffe2
