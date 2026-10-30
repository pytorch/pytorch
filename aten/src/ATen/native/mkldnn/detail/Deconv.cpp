#include <ATen/native/mkldnn/detail/Conv.h>
#include <ATen/native/mkldnn/detail/Deconv.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

namespace at::native::mkldnn {

namespace {

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

// Ported from ideep::convolution_transpose_forward::get_primitive_desc. The
// descriptor comes from ideep's primitive cache, shared with ideep's own
// convolution_transpose_forward.
template <bool with_bias>
dnnl::deconvolution_forward::primitive_desc get_primitive_desc(
    const ideep::tensor::desc& src_desc,
    const ideep::tensor::desc& weights_desc,
    const ideep::tensor::desc& bias_desc,
    const ideep::tensor::desc& dst_desc,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::deconvolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc_query = src_desc.to_format_any();
  auto weights_desc_query = weights_desc.to_format_any();
  auto bias_desc_query =
      with_bias ? bias_desc.to_format_any() : ideep::tensor::desc();
  auto dst_desc_query = dst_desc.to_format_any();

  // For nhwc path, weight uses format_tag::any,
  // while activation uses format_tag::nhwc.
  auto ndims = src_desc.get_dims().size();
  if (is_channels_last) {
    auto memory_format = dnnl::memory::format_tag::nhwc;
    if (3 == ndims) {
      memory_format = dnnl::memory::format_tag::nwc;
    } else if (5 == ndims) {
      memory_format = dnnl::memory::format_tag::ndhwc;
    }
    src_desc_query = src_desc.to_format(memory_format);
    dst_desc_query = dst_desc.to_format(memory_format);
  }
  auto key = ideep::utils::create_key(
      aprop_kind,
      aalgorithm,
      src_desc_query,
      weights_desc_query,
      bias_desc_query,
      dst_desc_query,
      with_bias,
      strides,
      dilates,
      padding_l,
      padding_r,
      attr,
      omp_get_max_threads());
  return ideep::utils::computation_cache<
      dnnl::deconvolution_forward::primitive_desc>::
      fetch_or_create(key, [&]() {
        if (with_bias) {
          return dnnl::deconvolution_forward::primitive_desc(
              engine,
              aprop_kind,
              aalgorithm,
              src_desc_query,
              weights_desc_query,
              bias_desc_query,
              dst_desc_query,
              strides,
              dilates,
              padding_l,
              padding_r,
              attr);
        } else {
          return dnnl::deconvolution_forward::primitive_desc(
              engine,
              aprop_kind,
              aalgorithm,
              src_desc_query,
              weights_desc_query,
              dst_desc_query,
              strides,
              dilates,
              padding_l,
              padding_r,
              attr);
        }
      });
}

// Ported from ideep::convolution_transpose_forward::do_prepare.
// For fp32
template <bool with_bias>
void do_prepare(
    deconv_forward_params& param,
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
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  ideep::tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
  ideep::attr_t op_attr, src_attr, weights_attr, bias_attr;
  ideep::tensor weights_grouped;
  dnnl::memory::dims dil_compatible;

  prepare_parameters(
      src,
      weights,
      bias,
      dst_dims,
      dst,
      dilates,
      groups,
      attr,
      with_bias,
      /*is_deconv=*/true,
      weights_grouped,
      dil_compatible,
      op_attr,
      src_attr,
      weights_attr,
      bias_attr,
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc);

  param.pd = get_primitive_desc<with_bias>(
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc,
      strides,
      dil_compatible,
      padding_l,
      padding_r,
      is_channels_last,
      op_attr,
      aalgorithm,
      aprop_kind);

  param.primitive = dnnl::deconvolution_forward(param.pd);
  param.op_attr = std::move(op_attr);
  param.bias_attr = std::move(bias_attr);
  param.groups = groups;
}

// Ported from ideep::convolution_transpose_forward::do_compute.
// For fp32 and int8
// `reorder_src` and `reorder_weight` indicate whether
// src/dst and weight/bias should be checked and possibly reordered
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute(
    const deconv_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto scratchpad = ideep::tensor(param.pd.scratchpad_desc());
  auto& expected_src =
      reorder_src ? src.reorder_if_differ_in(param.pd.src_desc()) : src;
  auto&& grouped_weights =
      weights.make_grouped_weights(param.groups, /*is_deconv=*/true);
  auto&& expected_weights = reorder_weight
      ? grouped_weights.reorder_if_differ_in(param.pd.weights_desc())
      : grouped_weights;
  if (reorder_src) {
    dst.reinit_if_possible(param.pd.dst_desc());
  }
  auto& primitive = param.primitive;

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, bias});
  }
  // Insert tensors of scales and zero points to args
  if (param.all_scales && !param.all_scales->empty()) {
    for (auto& arg_scale_pair : *param.all_scales) {
      int dnnl_arg = arg_scale_pair.first;
      ideep::tensor& scales_m = arg_scale_pair.second;
      args.insert({DNNL_ARG_ATTR_SCALES | dnnl_arg, scales_m});
    }
  }
  if (param.all_zero_points && !param.all_zero_points->empty()) {
    for (auto& arg_zp_pair : *param.all_zero_points) {
      int dnnl_arg = arg_zp_pair.first;
      ideep::tensor& zp_m = arg_zp_pair.second;
      args.insert({DNNL_ARG_ATTR_ZERO_POINTS | dnnl_arg, zp_m});
    }
  }
  primitive.execute(ideep::stream::default_stream(), args);
}

// Ported from ideep::convolution_transpose_forward::compute_impl.
// For 2-in-1 compute. For fp32
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_impl(
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
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  deconv_forward_params param;
  do_prepare<with_bias>(
      param,
      src,
      weights,
      bias,
      dst_dims,
      dst,
      strides,
      dilates,
      padding_l,
      padding_r,
      groups,
      is_channels_last,
      attr,
      aalgorithm,
      aprop_kind);
  do_compute<with_bias, reorder_src, reorder_weight>(
      param, src, weights, bias, dst);
}

// Ported from ideep::convolution_transpose_backward_weights::compute_impl.
template <bool with_diff_bias>
void backward_weights_compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims, // [i, o, ...]
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm) {
  const auto& engine = ideep::engine::cpu_engine();
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  // make diff_weights and dilates compatible with DNNL
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates);

  // dim: [i, o, ...]
  auto diff_weights_desc = ideep::tensor::desc(
      diff_weights_dims,
      diff_dst.get_data_type(),
      dnnl::memory::format_tag::any);

  if (groups > 1) {
    // dim: [g, o, i/g, ...]
    diff_weights_desc = diff_weights_desc.to_grouped(groups).transpose(1, 2);
  } else {
    // dim: [o, i, ...]
    diff_weights_desc = diff_weights_desc.transpose(0, 1);
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

  auto diff_bias_desc = with_diff_bias
      ? ideep::tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
            .to_format_any()
      : ideep::tensor::desc();

  auto op_attr = attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto forward_hints = get_primitive_desc<with_diff_bias>(
      src_desc,
      diff_weights_desc,
      diff_bias_desc,
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last,
      op_attr,
      aalgorithm,
      dnnl::prop_kind::forward);

  auto pd = with_diff_bias
      ? dnnl::deconvolution_backward_weights::primitive_desc(
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
      : dnnl::deconvolution_backward_weights::primitive_desc(
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
    dnnl::deconvolution_backward_weights(pd).execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
         {DNNL_ARG_DIFF_BIAS, diff_bias},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  } else {
    dnnl::deconvolution_backward_weights(pd).execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }

  diff_weights.feed_from(expected_diff_weights, /*is_deconv_weights=*/true);

  // recover output dims to align with pytorch
  if (groups > 1) {
    // [g, o, i/g, ...] -> [g, i/g, o, ...]
    diff_weights.transpose_(1, 2);
  } else {
    // [o, i, ...] -> [i, o, ...]
    diff_weights.transpose_(0, 1);
  }
}

} // namespace

template <bool reorder_src, bool reorder_weight>
void convolution_transpose_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  if (bias.is_empty()) {
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src,
        weights,
        bias,
        dst_dims,
        dst,
        strides,
        dilates,
        padding_l,
        padding_r,
        groups,
        is_channels_last,
        attr,
        aalgorithm,
        aprop_kind);
  } else {
    compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
        src,
        weights,
        bias,
        dst_dims,
        dst,
        strides,
        dilates,
        padding_l,
        padding_r,
        groups,
        is_channels_last,
        attr,
        aalgorithm,
        aprop_kind);
  }
}

template <bool reorder_src, bool reorder_weight>
void convolution_transpose_forward_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const dnnl::memory::dims& dst_dims,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  static const ideep::tensor dummy_bias;
  compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
      src,
      weights,
      dummy_bias,
      dst_dims,
      dst,
      strides,
      dilates,
      padding_l,
      padding_r,
      groups,
      is_channels_last,
      attr,
      aalgorithm,
      aprop_kind);
}

template <bool is_channels_last, bool transposed>
ideep::tensor::desc convolution_transpose_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims, // [i, o, ...]
    dnnl::memory::data_type dtype,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    int groups,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind,
    const dnnl::memory::dims& src_dims,
    const ideep::attr_t& attr) {
  auto src_size =
      weights_dims.size(); // weights_dims is 4 for conv2d and 5 for conv3d
  auto grouped = groups > 1;
  auto weights_dims_g = grouped
      ? ideep::utils::group_dims(weights_dims, groups)
      : weights_dims;
  // (g)iohw -> (g)oihw
  std::swap(weights_dims_g[grouped + 0], weights_dims_g[grouped + 1]);
  auto weights_desc = ideep::tensor::desc(weights_dims_g, dtype);

  auto dims_in = weights_desc.get_dims();
  auto ndims = dims_in.size();
  auto g = grouped ? dims_in[0] : 1;
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates);

  dnnl::memory::dims x_dims, y_dims, kernel_size;
  auto ic = g * dims_in[1 + grouped];
  auto oc = g * dims_in[0 + grouped];
  if (5 == src_size) {
    kernel_size.push_back(dims_in[ndims - 3]);
  }
  kernel_size.push_back(dims_in[ndims - 2]);
  kernel_size.push_back(dims_in[ndims - 1]);
  if (src_dims.empty()) {
    // Construct a dummy case
    x_dims.push_back(1);
    x_dims.push_back(ic);
    y_dims.push_back(1);
    y_dims.push_back(oc);
    auto valid_x_dim = [=](int idx) {
      return std::max(
          (padding_l[idx] + padding_r[idx] -
           (1 + (kernel_size[idx] - 1) * dilates[idx])) /
                  strides[idx] +
              2,
          2 * kernel_size[idx]);
    };
    if (4 == src_size) {
      x_dims.push_back(valid_x_dim(0));
      x_dims.push_back(valid_x_dim(1));
    } else {
      x_dims.push_back(valid_x_dim(0));
      x_dims.push_back(valid_x_dim(1));
      x_dims.push_back(valid_x_dim(2));
    }
  } else {
    // Use the real data
    for (size_t i = 0; i < src_size; ++i) {
      x_dims.push_back(src_dims[i]);
    }
    y_dims.push_back(src_dims[0]);
    y_dims.push_back(oc);
  }
  for (size_t d = 2; d < src_size; ++d) {
    auto out_size = (x_dims[d] - 1) * strides[d - 2] +
        (1 + (kernel_size[d - 2] - 1) * (dilates[d - 2])) - padding_l[d - 2] -
        padding_r[d - 2];
    y_dims.push_back(out_size);
  }
  auto x_dtype = (dtype != dnnl::memory::data_type::s8)
      ? dtype
      : dnnl::memory::data_type::u8;
  auto y_dtype = (dtype != dnnl::memory::data_type::s8)
      ? dtype
      : dnnl::memory::data_type::s32;

  ideep::tensor::desc src_desc(x_dims, x_dtype);
  ideep::tensor::desc dst_desc(y_dims, y_dtype);

  auto pd = get_primitive_desc</*with_bias=*/false>(
      src_desc,
      weights_desc,
      ideep::tensor::desc(),
      dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last,
      attr,
      aalgorithm,
      aprop_kind);

  // embed group info into weights_desc
  if (grouped) {
    if (transposed) {
      // [g, o, i/g, ...] -> [g, i/g, o, ...]
      return ideep::tensor::desc(pd.weights_desc(), groups).transpose(1, 2);
    } else {
      return ideep::tensor::desc(pd.weights_desc(), groups);
    }
  } else {
    if (transposed) {
      // [o, i, ...] -> [i, o, ...]
      return ideep::tensor::desc(pd.weights_desc(), groups).transpose(0, 1);
    } else {
      return ideep::tensor::desc(pd.weights_desc(), groups);
    }
  }
}

void convolution_transpose_backward_data_compute_v3(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    const int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm) {
  const auto& engine = ideep::engine::cpu_engine();
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
  auto weights_ = weights.make_grouped_weights(groups, true);
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates);

  // bool is_nhwc = diff_dst.get_desc().is_nhwc();
  // bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
  // auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
  // auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
  // auto weights_desc = weights_.get_desc().to_format_any();
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

  auto weights_desc =
      weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

  auto diff_src_desc = ideep::tensor::desc(
      diff_src_dims, diff_dst_desc.get_data_type(), format_tag);
  auto op_attr = attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto forward_hints = get_primitive_desc</*with_bias=*/false>(
      diff_src_desc,
      weights_desc,
      ideep::tensor::desc(),
      diff_dst_desc,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last,
      op_attr);

  auto pd = dnnl::deconvolution_backward_data::primitive_desc(
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

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::deconvolution_backward_data(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
       {DNNL_ARG_WEIGHTS, expected_weights},
       {DNNL_ARG_DIFF_SRC, diff_src},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

void convolution_transpose_backward_weights_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    const int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm) {
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
      attr,
      aalgorithm);
}

void convolution_transpose_backward_weights_compute_v3(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    const int groups,
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm) {
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
      attr,
      aalgorithm);
}

// The template arguments used by ATen callers.
template void convolution_transpose_forward_compute_v3<true, true>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    const dnnl::memory::dims&,
    ideep::tensor&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    int,
    bool,
    const ideep::attr_t&,
    dnnl::algorithm,
    dnnl::prop_kind);

template void convolution_transpose_forward_compute_v3<true, true>(
    const ideep::tensor&,
    const ideep::tensor&,
    const dnnl::memory::dims&,
    ideep::tensor&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    int,
    bool,
    const ideep::attr_t&,
    dnnl::algorithm,
    dnnl::prop_kind);

template ideep::tensor::desc
convolution_transpose_forward_expected_weights_desc<true, true>(
    const dnnl::memory::dims&,
    dnnl::memory::data_type,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    int,
    dnnl::algorithm,
    dnnl::prop_kind,
    const dnnl::memory::dims&,
    const ideep::attr_t&);

template ideep::tensor::desc
convolution_transpose_forward_expected_weights_desc<false, true>(
    const dnnl::memory::dims&,
    dnnl::memory::data_type,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    int,
    dnnl::algorithm,
    dnnl::prop_kind,
    const dnnl::memory::dims&,
    const ideep::attr_t&);

template ideep::tensor::desc
convolution_transpose_forward_expected_weights_desc<true, false>(
    const dnnl::memory::dims&,
    dnnl::memory::data_type,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    const dnnl::memory::dims&,
    int,
    dnnl::algorithm,
    dnnl::prop_kind,
    const dnnl::memory::dims&,
    const ideep::attr_t&);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
