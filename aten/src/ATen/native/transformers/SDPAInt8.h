#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/SDPAInt8.h>

namespace at::native {

at::Tensor sdpa_int8_math_impl(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    c10::optional<at::Tensor> attn_mask_,
    c10::optional<double> scale,
    double dropout_p,
    bool is_causal,
    int32_t q_zp,
    float q_scale,
    int32_t k_zp,
    float k_scale,
    int32_t v_zp,
    float v_scale,
    int32_t a_zp,
    float a_scale,
    int32_t o_zp,
    float o_scale);

using sdpa_int8_fn = void (*)(
    Tensor& output,
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
    double dropout_p, bool is_causal,
    int64_t q_zp,
    double q_scale,
    int64_t k_zp,
    double k_scale,
    int64_t v_zp,
    double v_scale,
    int64_t a_zp,
    double a_scale,
    int64_t o_zp,
    double o_scale);

DECLARE_DISPATCH(sdpa_int8_fn, sdpa_int8_kernel);

} // namespace at::native
