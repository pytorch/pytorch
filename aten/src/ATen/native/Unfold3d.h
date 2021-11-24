#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using unfold3d_fn = void (*)(
    ScalarType dtype,
    const void *src,
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
    void *dst
);

DECLARE_DISPATCH(unfold3d_fn, unfolded3d_copy_stub);
DECLARE_DISPATCH(unfold3d_fn, unfolded3d_acc_stub);

} // namespace native
} // namespace at
