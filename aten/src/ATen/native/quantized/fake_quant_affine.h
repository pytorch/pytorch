#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using fake_quant_tensor_cachemask_fn = void (*)(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

using fake_quant_tensor_cachemask_tensor_qparams_fn = void (*)(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& sc,
    const Tensor& z_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max);

using fake_quant_learnable_grad_tensor_fn = void (*)(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);

DECLARE_DISPATCH(fake_quant_tensor_cachemask_fn, fake_quant_tensor_cachemask_stub);
DECLARE_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_fn, fake_quant_tensor_cachemask_tensor_qparams_stub);
DECLARE_DISPATCH(fake_quant_learnable_grad_tensor_fn, fake_quant_grad_learnable_tensor_stub);

using fake_quant_per_channel_fn = void (*)(
    TensorIterator &iter,
    int64_t quant_min,
    int64_t quant_max);

using fake_quant_per_channel_cachemask_fn = void (*)(
    TensorIterator &iter,
    TensorIterator &iter_mask,
    int64_t quant_min,
    int64_t quant_max);

DECLARE_DISPATCH(fake_quant_per_channel_cachemask_fn, fake_quant_per_channel_cachemask_stub);

using fake_quant_learnable_per_channel_fn = void (*)(
    TensorIterator &iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);

DECLARE_DISPATCH(fake_quant_learnable_per_channel_fn, fake_quant_grad_learnable_channel_stub);

} // namespace native
} // namespace at
