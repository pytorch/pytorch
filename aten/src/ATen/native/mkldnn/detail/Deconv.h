#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::deconv_forward_params.
struct deconv_forward_params {
  deconv_forward_params() {}

  deconv_forward_params(
      dnnl::deconvolution_forward::primitive_desc&& pd,
      dnnl::deconvolution_forward&& primitive,
      int groups,
      ideep::attr_t&& op_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        groups(groups),
        op_attr(std::move(op_attr)),
        bias_attr(ideep::attr_t()) {}

  deconv_forward_params(
      dnnl::deconvolution_forward::primitive_desc&& pd,
      dnnl::deconvolution_forward&& primitive,
      int groups,
      ideep::attr_t&& op_attr,
      ideep::attr_t&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        groups(groups),
        op_attr(std::move(op_attr)),
        bias_attr(std::move(bias_attr)) {}

  dnnl::deconvolution_forward::primitive_desc pd;
  dnnl::deconvolution_forward primitive;
  int groups;
  ideep::attr_t op_attr;
  ideep::attr_t bias_attr;
  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_zero_points =
      nullptr;
};

// Ported from ideep::convolution_transpose_forward::compute (int8 2-in-1).
// 2-in-1 Compute for int8
// With bias. Bias is disabled if it is empty.
// This one does not have `is_channels_last` argument for compatibility
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute(
    const ideep::tensor& src,
    const ideep::tensor& weights, // dim: {o, i[, d], h, w}
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    int groups,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_transpose_forward::prepare (int8).
// Prepare for int8
// With bias. Bias is disabled if it is empty.
// This one does not have `is_channels_last` argument for compatibility
void convolution_transpose_forward_prepare(
    deconv_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights, // dim: {o, i[, d], h, w}
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates, // default = {1, 1}
    int groups, // default = 1
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_transpose_forward::compute (prepared params).
// Compute with prepared params. For both fp32 and int8
// With bias. Bias is disabled if it is empty.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Set them to False if you are sure they don't need reordering.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute(
    const deconv_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst);

// Ported from ideep::convolution_transpose_forward::compute_v3 (with bias).
// 2-in-1 Compute for fp32
// With bias. Bias is disabled if it is empty.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Set them to False if you are sure they don't need reordering.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights, // dim: {o, i[, d], h, w}
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_transpose_forward::compute_v3 (without bias).
// 2-in-1 Compute for fp32
// Without bias.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Set them to False if you are sure they don't need reordering.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights, // dim: {o, i[, d], h, w}
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_transpose_forward::expected_weights_desc.
/// @param is_channels_last Indicate whether weight is channels-last or not.
/// @param transposed If true, return weight desc in [i, o, ...] or [g, i/g, o, ...]
///                   format, which is used by PyTorch/Aten. If false, return desc in
///                   [o, i, ...] or [g, o, i/g, ...] format used by oneDNN. By default
///                   it is true for backward compatibility (e.g. Caffe2)
template <bool is_channels_last = false, bool transposed = true>
ideep::tensor::desc convolution_transpose_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims, // [i, o, ...]
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    const dnnl::memory::dims& strides = {1, 1},
    const dnnl::memory::dims& padding_l = {0, 0},
    const dnnl::memory::dims& padding_r = {0, 0},
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::convolution_transpose_backward_data::compute_v3.
void convolution_transpose_backward_data_compute_v3(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights, // dim: {i, o[, d], h, w}
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

// Ported from ideep::convolution_transpose_backward_weights::compute_v3 (with
// bias).
void convolution_transpose_backward_weights_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

// Ported from ideep::convolution_transpose_backward_weights::compute_v3
// (without bias).
void convolution_transpose_backward_weights_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
