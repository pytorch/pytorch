#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

class TensorIterator;

namespace at {

namespace native {

using fake_quant_tensor_fn = void (*)(
    Tensor& output,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

using fake_quant_grad_tensor_fn = void (*)(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

DECLARE_DISPATCH(fake_quant_tensor_fn, fake_quant_tensor_stub);
DECLARE_DISPATCH(fake_quant_grad_tensor_fn, fake_quant_grad_tensor_stub);

using fake_quant_by_channel_fn = void (*)(
    TensorIterator &iter,
    int64_t quant_min,
    int64_t quant_max);

DECLARE_DISPATCH(fake_quant_by_channel_fn, fake_quant_by_channel_stub);
DECLARE_DISPATCH(fake_quant_by_channel_fn, fake_quant_grad_by_channel_stub);

} // namespace native
} // namespace at
