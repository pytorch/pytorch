#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <memory>
#include <unordered_map>
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
    const Attr& attr,
    bool with_bias,
    bool is_deconv,
    ideep::tensor& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    Attr& op_attr, /* Output */
    Attr& src_attr, /* Output */
    Attr& weights_attr, /* Output */
    Attr& bias_attr, /* Output */
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
    const Attr& attr,
    const ideep::lowp_kind alowp_kind,
    bool with_bias,
    bool is_deconv,
    ideep::tensor& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    Attr& op_attr, /* Output */
    Attr& src_attr, /* Output */
    Attr& weights_attr, /* Output */
    Attr& bias_attr, /* Output */
    ideep::tensor::desc& src_desc, /* Output */
    ideep::tensor::desc& weights_desc, /* Output */
    ideep::tensor::desc& bias_desc, /* Output */
    ideep::tensor::desc& dst_desc /* Output */);

// Ported from ideep::convolution_forward_params.
struct convolution_forward_params {
  convolution_forward_params() {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      Attr&& op_attr,
      int groups)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(Attr()),
        pd_use_threads(omp_get_max_threads()) {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      Attr&& op_attr,
      int groups,
      Attr&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(std::move(bias_attr)),
        pd_use_threads(omp_get_max_threads()) {}

  dnnl::convolution_forward::primitive_desc pd;
  dnnl::convolution_forward primitive;
  Attr op_attr;
  int groups;
  Attr bias_attr;
  // From IPEX. Set in `do_prepare()`, only used outside ideep.
  // TO-DO: Use a better name, i.e., num_threads
  int pd_use_threads;

  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_zero_points =
      nullptr;
};

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
    const Attr& attr = Attr());

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
    const Attr& attr = Attr());

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
    const Attr& attr = Attr(),
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
    const Attr& attr = Attr(),
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
    const Attr& attr = Attr(),
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
    const Attr& attr = Attr(),
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
    const Attr& attr = Attr(),
    bool is_channels_last = false);

// Ported from ideep::convolution_forward::prepare.
// Prepare for int8 with bias
// This one does not have `is_channels_last` argument for compatibility
void convolution_forward_prepare(
    convolution_forward_params& param,
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
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_forward::compute (pre-prepared params).
// Conv computation with pre-prepared params, with bias
//   param: pd, primitive, groups ...
// for both fp32 and int8
// reorder as true means reorder might needed for src/weight/bias/dst as
// adapt to oneDNN recommended memory format or your simply not sure, set
// to false if src/weight/bias/dst are pre-reorded or no reorder needed as
// you know for sure
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_forward_compute(
    const convolution_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst);

// Ported from ideep::convolution_forward::compute (int8 2-in-1).
// 2-in-1 compute for int8 without bias
// This one does not have `is_channels_last` argument for compatibility
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_forward_compute(
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
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const Attr& attr = Attr(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::convolution_backward_data::compute_v2.
void convolution_backward_data_compute_v2(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const int groups,
    bool is_channels_last = false,
    Attr op_attr = Attr());

// Ported from ideep::convolution_backward_weights::compute_v2 (with bias).
void convolution_backward_weights_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const Attr& attr = Attr());

// Ported from ideep::convolution_backward_weights::compute_v2 (without bias).
void convolution_backward_weights_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const Attr& attr = Attr());

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
