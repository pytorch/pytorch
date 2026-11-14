#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::conv_deconv_utils::prepare_parameters (shared with
// detail/Deconv.cpp).
// Common logic to prepare parameters for conv/deconv.
// non-quantization version
void prepare_parameters(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    const ideep::tensor& dst,
    const dnnl::memory::dims& dilates,
    int groups,
    const ideep::attr_t& attr,
    bool with_bias,
    bool is_deconv,
    ideep::tensor& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    ideep::attr_t& op_attr, /* Output */
    ideep::attr_t& src_attr, /* Output */
    ideep::attr_t& weights_attr, /* Output */
    ideep::attr_t& bias_attr, /* Output */
    ideep::tensor::desc& src_desc, /* Output */
    ideep::tensor::desc& weights_desc, /* Output */
    ideep::tensor::desc& bias_desc, /* Output */
    ideep::tensor::desc& dst_desc /* Output */);

// Ported from ideep::conv_deconv_utils::prepare_parameters (shared with
// detail/Deconv.cpp).
// Common logic to prepare parameters for conv/deconv
// quantization version
void prepare_parameters(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    const ideep::tensor& dst,
    const dnnl::memory::dims& dilates,
    int groups,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const ideep::attr_t& attr,
    const ideep::lowp_kind alowp_kind,
    bool with_bias,
    bool is_deconv,
    ideep::tensor& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    ideep::attr_t& op_attr, /* Output */
    ideep::attr_t& src_attr, /* Output */
    ideep::attr_t& weights_attr, /* Output */
    ideep::attr_t& bias_attr, /* Output */
    ideep::tensor::desc& src_desc, /* Output */
    ideep::tensor::desc& weights_desc, /* Output */
    ideep::tensor::desc& bias_desc, /* Output */
    ideep::tensor::desc& dst_desc /* Output */);

// Ported from ideep::convolution_forward::compute_v2 (with bias).
// DEPRECATED
// 2-in-1 compute (prepare & compute) with bias
// Bias is not used if it is empty.
// Zero points are passed explicitly as arguments for quantization
void convolution_forward_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    const std::vector<float>& src_scales = std::vector<float>(),
    const std::vector<float>& weights_scales = std::vector<float>(),
    const std::vector<float>& dst_scales = std::vector<float>(),
    const std::vector<int32_t>& src_zero_point = std::vector<int32_t>(),
    const std::vector<int32_t>& dst_zero_point = std::vector<int32_t>(),
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::convolution_forward::compute_v2 (without bias).
// DEPRECATED
// 2-in-1 compute (prepare & compute) without bias
// Zero points are passed explicitly as arguments for quantization
void convolution_forward_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    const std::vector<float>& src_scales = std::vector<float>(),
    const std::vector<float>& weights_scales = std::vector<float>(),
    const std::vector<float>& dst_scales = std::vector<float>(),
    const std::vector<int32_t>& src_zero_point = std::vector<int32_t>(),
    const std::vector<int32_t>& dst_zero_point = std::vector<int32_t>(),
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::convolution_forward::compute_v3 (with bias).
// 2-in-1 Conv computation with bias for fp32
void convolution_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_forward::compute_v3 (without bias).
// 2-in-1 Conv computation w/o bias for fp32
void convolution_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_forward::compute_binary (with bias).
// 2-in-1 compute (prepare & compute) for fp32 with binary fusion
// With bias. Bias is not used if it is empty.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_forward_compute_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_forward::compute_binary (without bias).
// 2-in-1 compute (prepare & compute) for fp32 with binary fusion
// Without bias.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_forward_compute_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_forward::expected_weights_desc (the overload
// taking is_channels_last as an argument).
ideep::tensor::desc convolution_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    const dnnl::memory::dims& strides = {1, 1},
    const dnnl::memory::dims& padding_l = {0, 0},
    const dnnl::memory::dims& padding_r = {0, 0},
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    dnnl::memory::data_type x_dtype = dnnl::memory::data_type::f32,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    const ideep::attr_t& attr = ideep::attr_t(),
    bool is_channels_last = false);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
