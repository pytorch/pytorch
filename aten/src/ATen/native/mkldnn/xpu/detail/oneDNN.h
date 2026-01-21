#pragma once

#include <ATen/ATen.h>
#include <ATen/BlasBackend.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

namespace at::native::onednn {

TORCH_API sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event convolution_backward_weights(
    at::Tensor& diff_weight,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_weight_aten_size,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event convolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event deconvolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event deconvolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps = {});

TORCH_API sycl::event deconvolution_backward_weights(
    at::Tensor& diff_weight,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps = {});

TORCH_API void woq_matmul_int4(
    at::Tensor& result, // dst, [M, N]
    const at::Tensor& mat1_, // src, [M, K]
    const at::Tensor& mat2_, // quantized weight, [K/8, N]
    const at::Tensor& scale, // [K/group_size, N]
    const at::Tensor& zp, // [k/group_size, N]
    int64_t group_size,
    bool pri_cache = true);

dnnl::memory::dims conv_dst_size(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation);

dnnl::memory::dims deconv_dst_size(
    IntArrayRef src_size,
    IntArrayRef wgh_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef dst_padding,
    int64_t groups);

at::Tensor quantized_convolution(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    at::Tensor output,
    double inv_output_scale,
    int64_t output_zero_point,
    std::optional<at::Tensor> accum,
    double accum_scale,
    int64_t accum_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<std::string_view> binary_attr,
    std::optional<at::Scalar> binary_alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm);

void quantized_matmul(
    at::Tensor mat1, // act
    double input_scale,
    int64_t input_zero_point,
    at::Tensor mat2, // weight
    at::Tensor& weight_scales,
    at::Tensor& weight_zero_points,
    at::Tensor& b_raw,
    at::Tensor result, // output
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<at::Tensor> other, // extra input for binary-post-op
    double other_scale,
    int64_t other_zero_point,
    const std::string_view& binary_post_op,
    double binary_alpha,
    const std::string_view& unary_post_op,
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    std::string_view unary_post_op_algorithm,
    bool m2_trnas);

void sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_kv,
    int num_head_q,
    int num_head_kv,
    int head_dim_qk,
    int head_dim_v,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    float softmax_scale,
    const Tensor& attention,
    bool compute_logsumexp,
    const Tensor& logsumexp);

void sdpa_backward(
    int batch_size,
    int num_head_q,
    int num_head_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_qk,
    int head_dim_v,
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    float softmax_scale,
    Tensor& grad_query,
    Tensor& grad_key,
    Tensor& grad_value);

sycl::event scaled_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result,
    const Tensor& scale_a,
    const Tensor& scale_b,
    at::blas::ScalingType scaling_choice_a,
    at::blas::ScalingType scaling_choice_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    bool use_fast_accum);
} // namespace at::native::onednn
