#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::deconv_forward_params.
struct deconv_forward_params {
  deconv_forward_params() {}

  dnnl::deconvolution_forward::primitive_desc pd;
  dnnl::deconvolution_forward primitive;
  int groups;
  Attr op_attr;
  Attr bias_attr;
  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, MkldnnMemory>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, MkldnnMemory>> all_zero_points =
      nullptr;
};

// Ported from ideep::convolution_transpose_forward::compute (int8 2-in-1).
// 2-in-1 Compute for int8
// With bias. Bias is disabled if it is empty.
// Channels-last is inferred from the src and weights descriptors.
// Only <true, true> is instantiated.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute(
    const MkldnnMemory& src,
    const MkldnnMemory& weights, // dim: {o, i[, d], h, w}
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_transpose_forward::prepare (int8).
// Prepare for int8
// With bias. Bias is disabled if it is empty.
// Channels-last is inferred from the src and weights descriptors.
void convolution_transpose_forward_prepare(
    deconv_forward_params& param,
    const MkldnnMemory& src,
    const MkldnnMemory& weights, // dim: {o, i[, d], h, w}
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_transpose_forward::compute (prepared params).
// Compute with params from convolution_transpose_forward_prepare.
// With bias. Bias is disabled if it is empty.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Only <false, false> is instantiated.
template <bool reorder_src, bool reorder_weight>
void convolution_transpose_forward_compute(
    const deconv_forward_params& param,
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    MkldnnMemory& dst);

// Ported from ideep::convolution_transpose_forward::compute_v3 (with bias).
// 2-in-1 Compute for fp32
// With bias. Bias is disabled if it is empty.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Only <true, true> is instantiated.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute_v3(
    const MkldnnMemory& src,
    const MkldnnMemory& weights, // dim: {o, i[, d], h, w}
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    bool is_channels_last = false,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_transpose_forward::compute_v3 (without bias).
// 2-in-1 Compute for fp32
// Without bias.
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered.
// Only <true, true> is instantiated.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_transpose_forward_compute_v3(
    const MkldnnMemory& src,
    const MkldnnMemory& weights, // dim: {o, i[, d], h, w}
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    int groups = 1,
    bool is_channels_last = false,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward);

// Ported from ideep::convolution_transpose_forward::expected_weights_desc.
/// @param is_channels_last Whether src/dst use a channels-last format. Weights
///                   always use format_tag::any.
/// @param transposed If true, return weight desc in [i, o, ...] or [g, i/g, o, ...]
///                   format, which is used by PyTorch/Aten. If false, return desc in
///                   [o, i, ...] or [g, o, i/g, ...] format used by oneDNN. By default
///                   it is true, giving ATen's layout.
/// Instantiated for <true, true>, <false, true> and <true, false>.
template <bool is_channels_last = false, bool transposed = true>
MkldnnMemory::desc convolution_transpose_forward_expected_weights_desc(
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
    const Attr& attr = Attr());

// Ported from ideep::convolution_transpose_backward_data::compute_v3.
void convolution_transpose_backward_data_compute_v3(
    const MkldnnMemory& diff_dst,
    const MkldnnMemory& weights, // dim: {o, i[, d], h, w}
    const dnnl::memory::dims& diff_src_dims,
    MkldnnMemory& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

// Ported from ideep::convolution_transpose_backward_weights::compute_v3 (with
// bias).
void convolution_transpose_backward_weights_compute_v3(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    const dnnl::memory::dims& diff_weights_dims, // dim: {i, o[, d], h, w}
    MkldnnMemory& diff_weights, // dim: {i, o[, d], h, w}
    MkldnnMemory& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

// Ported from ideep::convolution_transpose_backward_weights::compute_v3
// (without bias).
void convolution_transpose_backward_weights_compute_v3(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    const dnnl::memory::dims& diff_weights_dims, // dim: {i, o[, d], h, w}
    MkldnnMemory& diff_weights, // dim: {i, o[, d], h, w}
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates = {1, 1},
    const int groups = 1,
    bool is_channels_last = false,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
