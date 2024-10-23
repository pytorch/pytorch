#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/SDPAInt8.h>

namespace at::native {

using sdpa_int8_fn = void (*)(
    Tensor& output,
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale,
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
