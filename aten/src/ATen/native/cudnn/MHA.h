#pragma once
#include <ATen/core/Tensor.h>

namespace at::native {

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset);

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t max_s_q,
    int64_t max_s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset);

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset);

} // namespace at::native
