#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <cstdint>

namespace at::native {

using unfold2d_fn = void (*)(
    ScalarType dtype,
    void *finput,
    void *input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last
);

DECLARE_DISPATCH(unfold2d_fn, unfolded2d_copy_stub);
DECLARE_DISPATCH(unfold2d_fn, unfolded2d_acc_stub);

} // namespace at::native
