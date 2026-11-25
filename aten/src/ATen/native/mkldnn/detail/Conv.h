#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::conv_deconv_utils::prepare_parameters (shared with
// detail/Deconv.cpp).
// Common logic to prepare parameters for conv/deconv.
// non-quantization version
void prepare_parameters(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    const MkldnnMemory& dst,
    const dnnl::memory::dims& dilates,
    int groups,
    const Attr& attr,
    bool with_bias,
    bool is_deconv,
    MkldnnMemory& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    Attr& op_attr, /* Output */
    Attr& src_attr, /* Output */
    Attr& weights_attr, /* Output */
    Attr& bias_attr, /* Output */
    MkldnnMemory::desc& src_desc, /* Output */
    MkldnnMemory::desc& weights_desc, /* Output */
    MkldnnMemory::desc& bias_desc, /* Output */
    MkldnnMemory::desc& dst_desc /* Output */);

// Ported from ideep::conv_deconv_utils::prepare_parameters (shared with
// detail/Deconv.cpp).
// Common logic to prepare parameters for conv/deconv
// quantization version. Also used for fp32 when scales and zero points are
// empty.
void prepare_parameters(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    const MkldnnMemory& dst,
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
    MkldnnMemory& weight_grouped, /* Output */
    dnnl::memory::dims& dil_compatible, /* Output */
    Attr& op_attr, /* Output */
    Attr& src_attr, /* Output */
    Attr& weights_attr, /* Output */
    Attr& bias_attr, /* Output */
    MkldnnMemory::desc& src_desc, /* Output */
    MkldnnMemory::desc& weights_desc, /* Output */
    MkldnnMemory::desc& bias_desc, /* Output */
    MkldnnMemory::desc& dst_desc /* Output */);

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
  // Set by the constructors but not read in ATen; kept to match ideep's layout.
  int pd_use_threads;

  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, MkldnnMemory>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, MkldnnMemory>> all_zero_points =
      nullptr;
};

// Ported from ideep::convolution_forward::compute_v2 (with bias).
// DEPRECATED
// 2-in-1 compute (prepare & compute) with bias
// Bias is not used if it is empty.
// Zero points are passed explicitly as arguments for quantization
void convolution_forward_compute_v2(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const MkldnnMemory& src,
    const MkldnnMemory& other,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
    const MkldnnMemory& src,
    const MkldnnMemory& other,
    const MkldnnMemory& weights,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
MkldnnMemory::desc convolution_forward_expected_weights_desc(
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

// Ported from ideep::convolution_backward_data::compute_v2.
// Signature matches ideep's (op_attr is taken by value and modified).
void convolution_backward_data_compute_v2(
    const MkldnnMemory& diff_dst,
    const MkldnnMemory& weights,
    const dnnl::memory::dims& diff_src_dims,
    MkldnnMemory& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const int groups,
    bool is_channels_last = false,
    Attr op_attr = Attr());

// Ported from ideep::convolution_backward_weights::compute_v2 (with bias).
void convolution_backward_weights_compute_v2(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    MkldnnMemory& diff_weights,
    MkldnnMemory& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const Attr& attr = Attr());

// Ported from ideep::convolution_backward_weights::compute_v2 (without bias).
void convolution_backward_weights_compute_v2(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    MkldnnMemory& diff_weights,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const Attr& attr = Attr());

// Ported from ideep::convolution_forward::prepare.
// Prepare params for int8 conv. Bias is not used if it is empty.
// Channels-last is inferred from the src and weights descriptors.
void convolution_forward_prepare(
    convolution_forward_params& param,
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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
// Conv computation with params from convolution_forward_prepare
//   param: pd, primitive, groups ...
// reorder as true means reorder might needed for src/weight/bias/dst as
// adapt to oneDNN recommended memory format or your simply not sure, set
// to false if src/weight/bias/dst are pre-reorded or no reorder needed as
// you know for sure
// Only <false, false> is instantiated.
template <bool reorder_src, bool reorder_weight>
void convolution_forward_compute(
    const convolution_forward_params& param,
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    MkldnnMemory& dst);

// Ported from ideep::convolution_forward::compute (int8 2-in-1).
// 2-in-1 compute for int8. Bias is not used if it is empty.
// Always passes is_channels_last = true, unlike convolution_forward_prepare,
// which infers it from src and weights.
template <bool reorder_src = true, bool reorder_weight = true>
void convolution_forward_compute(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    const dnnl::memory::dims& dst_dims,
    MkldnnMemory& dst,
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

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
