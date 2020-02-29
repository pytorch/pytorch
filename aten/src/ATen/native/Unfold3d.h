#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

void Unfold3dCopyCPU(
    const Tensor& src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    Tensor* dst);

void Unfold3dAccCPU(
    const Tensor& src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    Tensor* dst);

} // namespace native
} // namespace at
