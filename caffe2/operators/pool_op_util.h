#ifndef CAFFE2_OPERATORS_POOL_OP_UTIL_H_
#define CAFFE2_OPERATORS_POOL_OP_UTIL_H_

#include "caffe2/core/types.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {
namespace pool_op_util {

bool IsNeon4x4p0s0Eligible(
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    int pad_b,
    int pad_r,
    int dilation_h,
    int dilation_w,
    const float* X,
    float* Y);

bool IsNeon2x2p0s0Eligible(
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    int pad_b,
    int pad_r,
    int dilation_h,
    int dilation_w,
    const float* X,
    float* Y);

void RunNeonAveragePool4x4p0s0NCHW(
    int N,
    int C,
    int H,
    int W,
    const float* X,
    float* Y);

void RunNeonMaxPool2x2p0s0NCHW(
    int N,
    int C,
    int H,
    int W,
    const float* X,
    float* Y);

} // namespace pool_op_util
} // namespace caffe2

#endif // CAFFE2_OPERATORS_POOL_OP_UTIL_H_
