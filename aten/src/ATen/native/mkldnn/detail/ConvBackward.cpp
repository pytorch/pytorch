#include <ATen/native/mkldnn/detail/ConvBackward.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <unordered_map>

namespace at::native::mkldnn {

namespace {

// Ported from ideep::convolution_backward_weights::compute_impl.
template <bool with_diff_bias>
void backward_weights_compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const int groups,
    bool is_channels_last,
    const ideep::attr_t& attr) {
  const auto& engine = ideep::engine::cpu_engine();
  const auto aalgorithm = dnnl::algorithm::convolution_direct;
  dnnl::memory::data_type diff_dst_type = diff_dst.get_data_type();
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst_type,
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  // make diff_weights and dilates compatible with DNNL
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates);
  auto diff_weights_desc = ideep::tensor::desc(
      diff_weights_dims, diff_dst_type, dnnl::memory::format_tag::any);
  if (groups > 1) {
    diff_weights_desc = diff_weights_desc.to_grouped(groups).to_format_any();
  }

  auto format_tag = dnnl::memory::format_tag::any;
  auto ndims = diff_dst.get_desc().get_dims().size();
  if (ndims == 4) {
    if (is_channels_last) {
      format_tag = dnnl::memory::format_tag::nhwc;
    }
  } else if (ndims == 5) {
    if (is_channels_last) {
      format_tag = dnnl::memory::format_tag::ndhwc;
    }
  }
  auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
  auto src_desc = src.get_desc().to_format(format_tag);

  auto diff_bias_desc = ideep::tensor::desc(
      {diff_dst.get_dim(1)}, diff_dst_type, dnnl::memory::format_tag::any);

  // for forward hint, weights_desc should have same data_type
  // with other input desc, expect for bias_desc
  auto weights_desc = diff_weights_desc;
  auto op_attr = attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto forward_hints =
      ideep::convolution_forward::get_primitive_desc<with_diff_bias>(
      src_desc,
      weights_desc,
      diff_bias_desc,
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last,
      op_attr,
      aalgorithm,
      dnnl::prop_kind::forward,
      engine);

#ifdef __aarch64__
  auto pd = with_diff_bias
      ? dnnl::convolution_backward_weights::primitive_desc(
            engine,
            aalgorithm,
            src_desc,
            diff_weights_desc,
            diff_bias_desc,
            diff_dst_desc,
            strides,
            dilates_,
            padding_l,
            padding_r,
            forward_hints.first,
            op_attr)
      : dnnl::convolution_backward_weights::primitive_desc(
            engine,
            aalgorithm,
            src_desc,
            diff_weights_desc,
            diff_dst_desc,
            strides,
            dilates_,
            padding_l,
            padding_r,
            forward_hints.first,
            op_attr);
#else
  auto pd = with_diff_bias
      ? dnnl::convolution_backward_weights::primitive_desc(
            engine,
            aalgorithm,
            src_desc,
            diff_weights_desc,
            diff_bias_desc,
            diff_dst_desc,
            strides,
            dilates_,
            padding_l,
            padding_r,
            forward_hints,
            op_attr)
      : dnnl::convolution_backward_weights::primitive_desc(
            engine,
            aalgorithm,
            src_desc,
            diff_weights_desc,
            diff_dst_desc,
            strides,
            dilates_,
            padding_l,
            padding_r,
            forward_hints,
            op_attr);
#endif

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  // embed group info into diff_weights_desc
  auto expected_diff_weights_desc =
      ideep::tensor::desc(pd.diff_weights_desc(), groups);
  ideep::tensor expected_diff_weights;
  // diff_weights not init in FW or has same desc with expected desc.
  if (diff_weights.is_empty() ||
      diff_weights.get_desc() == expected_diff_weights_desc) {
    diff_weights.reinit_if_possible(expected_diff_weights_desc);
    expected_diff_weights = diff_weights;
  } else {
    expected_diff_weights.init(expected_diff_weights_desc);
  }

  ideep::tensor scratchpad(pd.scratchpad_desc());

  if (with_diff_bias) {
    diff_bias.reinit_if_possible(pd.diff_bias_desc());
    dnnl::convolution_backward_weights(pd).execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
         {DNNL_ARG_DIFF_BIAS, diff_bias},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  } else {
    dnnl::convolution_backward_weights(pd).execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
  // diff_weights has been init in FW side, but has diff desc with
  // expected_diff_weights.
  if (diff_weights.get_desc() != expected_diff_weights_desc) {
    if (!diff_weights.get_desc().has_same_shape_as(
            expected_diff_weights_desc)) {
      diff_weights.reinit_if_possible(expected_diff_weights_desc);
    }
    diff_weights.feed_from(expected_diff_weights);
  }
}
} // namespace

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
    bool is_channels_last,
    ideep::attr_t op_attr) {
  const auto& engine = ideep::engine::cpu_engine();
  const auto aalgorithm = dnnl::algorithm::convolution_direct;
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  // make weights and dilates compatible with DNNL
  auto weights_ = weights.make_grouped_weights(groups);
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates);
  auto format_tag = dnnl::memory::format_tag::any;
  auto ndims = diff_dst.get_desc().get_dims().size();
  if (ndims == 4) {
    if (is_channels_last) {
      format_tag = dnnl::memory::format_tag::nhwc;
    }
  } else if (ndims == 5) {
    if (is_channels_last) {
      format_tag = dnnl::memory::format_tag::ndhwc;
    }
  }
  auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
  // align weight data type with diff_dst for bf16 and f16
  auto weights_desc = ideep::tensor::desc(
      weights_.get_dims(),
      diff_dst.get_data_type(),
      dnnl::memory::format_tag::any);
  if (groups > 1) {
    weights_desc = weights_desc.to_grouped(groups);
  }

  auto diff_src_desc = ideep::tensor::desc(
      diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

  auto forward_hints =
      ideep::convolution_forward::get_primitive_desc</*with_bias=*/false>(
      diff_src_desc,
      weights_desc,
      ideep::tensor::desc(),
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last);

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef __aarch64__
  auto pd = dnnl::convolution_backward_data::primitive_desc(
      engine,
      aalgorithm,
      diff_src_desc,
      weights_desc,
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      forward_hints.first,
      op_attr);
#else
  auto pd = dnnl::convolution_backward_data::primitive_desc(
      engine,
      aalgorithm,
      diff_src_desc,
      weights_desc,
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      forward_hints,
      op_attr);
#endif

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());

  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::convolution_backward_data(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
       {DNNL_ARG_WEIGHTS, expected_weights},
       {DNNL_ARG_DIFF_SRC, diff_src},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

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
    bool is_channels_last,
    const ideep::attr_t& attr) {
  backward_weights_compute_impl</*with_diff_bias=*/true>(
      src,
      diff_dst,
      diff_weights_dims,
      diff_weights,
      diff_bias,
      strides,
      dilates,
      padding_l,
      padding_r,
      groups,
      is_channels_last,
      attr);
}

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
    bool is_channels_last,
    const ideep::attr_t& attr) {
  static ideep::tensor dummy_diff_bias;
  backward_weights_compute_impl</*with_diff_bias=*/false>(
      src,
      diff_dst,
      diff_weights_dims,
      diff_weights,
      dummy_diff_bias,
      strides,
      dilates,
      padding_l,
      padding_r,
      groups,
      is_channels_last,
      attr);
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
