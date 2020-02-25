#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

namespace native {

using fake_quant_slice_fn = void (*)(
    Tensor& output,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

using fake_quant_grad_slice_fn = void (*)(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

DECLARE_DISPATCH(fake_quant_slice_fn, fake_quant_slice_stub);
DECLARE_DISPATCH(fake_quant_grad_slice_fn, fake_quant_grad_slice_stub);

} // namespace native
} // namespace at
