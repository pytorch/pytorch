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

template <typename T, StorageOrder kOrder, bool kCountIncludePad>
void RunAveragePool1D(
    int N,
    int C,
    int X_size,
    int Y_size,
    int kernel,
    int stride,
    int pad,
    const T* X,
    T* Y);

template <typename T, StorageOrder kOrder, bool kCountIncludePad>
void RunAveragePool2D(
    int N,
    int C,
    int X_H,
    int X_W,
    int Y_H,
    int Y_W,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    const T* X,
    T* Y);

template <typename T, StorageOrder kOrder, bool kCountIncludePad>
void RunAveragePool3D(
    int N,
    int C,
    int X_D,
    int X_H,
    int X_W,
    int Y_D,
    int Y_H,
    int Y_W,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_p,
    int pad_t,
    int pad_l,
    const T* X,
    T* Y);

template <typename T, StorageOrder kOrder>
void RunMaxPool1D(
    const int N,
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* X,
    T* Y);

template <typename T, StorageOrder kOrder>
void RunMaxPool2D(
    int N,
    int C,
    int X_H,
    int X_W,
    int Y_H,
    int Y_W,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    const T* X,
    T* Y);

template <typename T, StorageOrder kOrder>
void RunMaxPool3D(
    int N,
    int C,
    int X_D,
    int X_H,
    int X_W,
    int Y_D,
    int Y_H,
    int Y_W,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_p,
    int pad_t,
    int pad_l,
    const T* X,
    T* Y);

} // namespace pool_op_util
} // namespace caffe2

#endif // CAFFE2_OPERATORS_POOL_OP_UTIL_H_
