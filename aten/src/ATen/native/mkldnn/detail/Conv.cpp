#include <ATen/native/mkldnn/detail/Conv.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

namespace at::native::mkldnn {

namespace {

// Ported from ideep::convolution_forward_params.
struct convolution_forward_params {
  convolution_forward_params() {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      ideep::attr_t&& op_attr,
      int groups)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(ideep::attr_t()),
        pd_use_threads(omp_get_max_threads()) {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      ideep::attr_t&& op_attr,
      int groups,
      ideep::attr_t&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(std::move(bias_attr)),
        pd_use_threads(omp_get_max_threads()) {}

  dnnl::convolution_forward::primitive_desc pd;
  dnnl::convolution_forward primitive;
  ideep::attr_t op_attr;
  int groups;
  ideep::attr_t bias_attr;
  // From IPEX. Set in `do_prepare()`, only used outside ideep.
  // TO-DO: Use a better name, i.e., num_threads
  int pd_use_threads;

  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_zero_points =
      nullptr;
};

// Ported from ideep::conv_deconv_utils::prepare_parameters.
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
    ideep::tensor::desc& dst_desc /* Output */) {
  std::vector<float> dst_scales_in;
  dnnl::memory::data_type dst_data_type;
  op_attr.set_post_ops(attr.get_post_ops());

  // make weights and dilates compatible with DNNL
  weight_grouped = weights.make_grouped_weights(groups, is_deconv);
  dil_compatible = ideep::utils::get_compatible_dilates(dilates);

  auto& weights_scales_in =
      weight_grouped.has_scale() ? weight_grouped.get_scale() : weights_scales;
  if (!weights_scales_in.empty()) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        alowp_kind == ideep::u8s8 || alowp_kind == ideep::s8s8
            ? dnnl_success
            : dnnl_invalid_arguments,
        "Unsupported lowp kind");
#endif
    int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
    auto& src_scales_in = src.has_scale()
        ? src.get_scale()
        : (src_scales.empty() ? ideep::IDEEP_DEF_SCALE : src_scales);

    // determine dst data type
    if (dst.get_data_type() != dnnl::memory::data_type::undef) {
      dst_data_type = dst.get_data_type();
    } else if (dst_scales.empty() || dst_scales == ideep::IDEEP_DEF_SCALE) {
      dst_data_type = dnnl::memory::data_type::f32;
    } else if (attr.non_negitive_output()) {
      dst_data_type = dnnl::memory::data_type::u8;
    } else {
      dst_data_type = dnnl::memory::data_type::s8;
    }

    // fill primitive attr
    dst_scales_in =
        dst_scales.empty() || dst_data_type == dnnl::memory::data_type::f32
        ? ideep::IDEEP_DEF_SCALE
        : dst_scales;
    const auto default_zero_point = std::vector<int32_t>(1);
    const auto& src_zero_point = src.has_zero_point() ? src.get_zero_point()
        : src_zero_points.empty()                     ? default_zero_point
                                                      : src_zero_points;
    const auto& dst_zero_point = [&]() {
      if (attr.has_op_kind(dnnl::primitive::kind::sum)) {
        // Similar logic as dst_scales_in. Since when fused with sum, the dst will be the tensor of sum,
        // In this case, the output tensor' dst_zero_points and dst_scales_in should be passed in explicitly.
#ifndef NDEBUG
        dnnl::error::wrap_c_api(
            !dst_zero_points.empty() ? dnnl_success : dnnl_invalid_arguments,
            "When conv fused with sum, dst_zero_points must be passed in.");
#endif
        return dst_zero_points;
      } else {
        // Keep the original logic for finding dst_zero_point when not fused with sum.
        return dst.has_zero_point() ? dst.get_zero_point()
            : dst_zero_points.empty() ? default_zero_point
                                      : dst_zero_points;
      }
    }();

    const auto src_zero_point_size =
        static_cast<dnnl::memory::dim>(src_zero_point.size());
    const auto dst_zero_point_size =
        static_cast<dnnl::memory::dim>(dst_zero_point.size());
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        src_zero_point_size == 1 && dst_zero_point_size == 1
            ? dnnl_success
            : dnnl_invalid_arguments,
        "DNNL only support 1-dim zero_point");
#endif

    if (attr.has_op_kind(dnnl::primitive::kind::sum)) {
      // Here we need to recalculate the scale of sum.
      // When fused with sum, dst_scales_in is the final output tensor's scale.
      // dst.scale is the scale of sum tensor.

      float sum_scale = 1.0 / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);

      // When fused with sum, the dst tensor is same as the sum tensor.
      // So the sum_zero_point will be fetched from the dst tensor.
      int32_t sum_zero_point =
          dst.has_zero_point() ? dst.get_zero_point()[0] : 0;
      if (attr.has_op_kind(dnnl::primitive::kind::eltwise)) {
        op_attr = ideep::attr_t::residual_with_sum_zero_point(
            sum_scale, sum_zero_point);
      } else {
        op_attr = ideep::attr_t::fuse_sum(sum_scale, sum_zero_point);
      }
    }
    // Set scales to op_attr
    if (src_scales_in[0] != 1.0f) {
      int src_scale_mask = 0;
      op_attr.set_scales(
          DNNL_ARG_SRC, src_scale_mask, {1.0f / src_scales_in[0]});
    }
    // For grouped conv, weight format is `goi...`, where g is groups and o and i are OC/IC per group.
    // `weight.get_dim(0)` returns g * o and `get_dim(1)` returns i.
    // For non-grouped conv (groups = 1), weight format is `oi...`.
    // `weight.get_dim(0)` returns o and `get_dim(1)` returns i.
    // So, it's always correct to get oc per group by `get_dim(0) / groups`
    // For deconv, weight format is `gio...` (grouped) or `io...` (non-grouped)
    // `weight.get_dim(0)` returns g * i or i and `get_dim(1)` always returns o.
    int oc_per_group = is_deconv ? weight_grouped.get_dim(1)
                                 : weight_grouped.get_dim(0) / groups;
    auto wei_scale_mask = ideep::utils::conv_weight_scale_mask(
        weights_scales_in.size(), oc_per_group, groups, is_deconv);
    auto wei_scales = weights_scales_in;
    if (!std::all_of(wei_scales.begin(), wei_scales.end(), [](float i) {
          return i == 1.0f;
        })) {
      op_attr.set_scales(DNNL_ARG_WEIGHTS, wei_scale_mask, wei_scales);
    }
    if (dst_scales_in[0] != 1.0f) {
      int dst_scale_mask = 0;
      op_attr.set_scales(
          DNNL_ARG_DST, dst_scale_mask, {1.0f / dst_scales_in[0]});
    }
    // Set zero points to op_attr (weight zero point is always 0. Do not set.)
    // Only set non-zero zero points otherwise there is extra oneDNN overhead
    if (src_zero_point[0] != 0) {
      auto src_zp_mask = ideep::utils::tensor_zp_mask(src_zero_point_size);
      op_attr.set_zero_points(DNNL_ARG_SRC, src_zp_mask, src_zero_point);
    }
    if (dst_zero_point[0] != 0) {
      auto dst_zp_mask = ideep::utils::tensor_zp_mask(dst_zero_point_size);
      op_attr.set_zero_points(DNNL_ARG_DST, dst_zp_mask, dst_zero_point);
    }

    src_desc = {
        src.get_dims(),
        alowp_kind == ideep::u8s8 ? dnnl::memory::data_type::u8
                                  : dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any};
    if (src.get_data_type() == dnnl::memory::data_type::f32) {
      src_attr = {0, src_scales_in};
    }

    weights_desc = ideep::tensor::desc(
        weight_grouped.get_dims(),
        dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any);
    if (groups > 1) {
      weights_desc = weights_desc.to_grouped(groups);
    }
    if (weight_grouped.get_data_type() == dnnl::memory::data_type::f32) {
      weights_attr = {
          ideep::utils::tensor_scale_mask(scale_size, groups > 1),
          weights_scales_in};
    }

    if (with_bias) {
      bias_desc = {
          bias.get_dims(),
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::any}; // Use f32 instead of s32 to improve accuracy
    }
  } else {
    if (src.has_scale()) {
      auto src_scale = src.get_scale();
      src_scale[0] = 1.0f / src_scale[0];
      src_attr = {0, src_scale};
    }

#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        one_of(
            weight_grouped.get_data_type(),
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::bf16)
            ? dnnl_success
            : dnnl_invalid_arguments,
        "Incorrect data type in weights");
#endif

    // align weights data type with src
    dst_data_type = src.get_data_type() == dnnl::memory::data_type::bf16
        ? dnnl::memory::data_type::bf16
        : dnnl::memory::data_type::f32;
    src_desc = src.get_desc().to_type(dst_data_type);
    weights_desc = ideep::tensor::desc(
        weight_grouped.get_dims(), dst_data_type, dnnl::memory::format_tag::any);
    if (groups > 1) {
      weights_desc = weights_desc.to_grouped(groups);
    }

    if (with_bias) {
#ifndef NDEBUG
      dnnl::error::wrap_c_api(
          one_of(
              bias.get_data_type(),
              dnnl::memory::data_type::f32,
              dnnl::memory::data_type::bf16)
              ? dnnl_success
              : dnnl_invalid_arguments,
          "Incorrect data type in bias");
#endif
      bias_desc = bias.get_desc();
    }
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dst_desc = attr.has_op_kind(dnnl::primitive::kind::sum)
      ? dst.get_desc()
      : ideep::tensor::desc(dst_dims, dst_data_type);
}

// Ported from ideep::conv_deconv_utils::prepare_parameters.
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
    ideep::tensor::desc& dst_desc /* Output */) {
  dnnl::memory::data_type dst_data_type;
  op_attr = attr;

  // make weights and dilates compatible with DNNL
  weight_grouped = weights.make_grouped_weights(groups, is_deconv);
  dil_compatible = ideep::utils::get_compatible_dilates(dilates);

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      one_of(
          weight_grouped.get_data_type(),
          dnnl::memory::data_type::f32,
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)
          ? dnnl_success
          : dnnl_invalid_arguments,
      "Incorrect data type in weights");
#endif

  // align weights data type with src
  dst_data_type = src.get_data_type() == dnnl::memory::data_type::bf16
      ? dnnl::memory::data_type::bf16
      : ((src.get_data_type() == dnnl::memory::data_type::f16)
             ? dnnl::memory::data_type::f16
             : dnnl::memory::data_type::f32);
  src_desc = src.get_desc().to_type(dst_data_type);
  weights_desc = ideep::tensor::desc(
      weight_grouped.get_dims(), dst_data_type, dnnl::memory::format_tag::any);
  if (groups > 1) {
    weights_desc = weights_desc.to_grouped(groups);
  }

  if (with_bias) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        one_of(
            bias.get_data_type(),
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::bf16,
            dnnl::memory::data_type::f16)
            ? dnnl_success
            : dnnl_invalid_arguments,
        "Incorrect data type in bias");
#endif
    bias_desc = bias.get_desc();
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dst_desc = attr.has_op_kind(dnnl::primitive::kind::sum)
      ? dst.get_desc()
      : ideep::tensor::desc(dst_dims, dst_data_type);
}

// Ported from ideep::convolution_forward::get_primitive_desc. On x86 the
// descriptor comes from ideep's primitive cache, shared with ideep's own
// convolution_forward.
#ifdef __aarch64__
template <bool with_bias>
std::pair<
    dnnl::convolution_forward::primitive_desc,
    dnnl::convolution_forward>
get_primitive_desc(
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
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc_query = src_desc;
  auto weights_desc_query = weights_desc;
  auto bias_desc_query = with_bias ? bias_desc : ideep::tensor::desc();
  auto dst_desc_query = dst_desc;

  src_desc_query = src_desc.to_format_any();
  weights_desc_query = weights_desc.to_format_any();
  bias_desc_query =
      with_bias ? bias_desc.to_format_any() : ideep::tensor::desc();
  dst_desc_query = dst_desc.to_format_any();

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

  dnnl::convolution_forward::primitive_desc pd;
  if (with_bias) {
    pd = dnnl::convolution_forward::primitive_desc(
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
    pd = dnnl::convolution_forward::primitive_desc(
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
  return {pd, dnnl::convolution_forward(pd)};
}
#else
template <bool with_bias>
dnnl::convolution_forward::primitive_desc get_primitive_desc(
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
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc_query = src_desc;
  auto weights_desc_query = weights_desc;
  auto bias_desc_query = with_bias ? bias_desc : ideep::tensor::desc();
  auto dst_desc_query = dst_desc;

  src_desc_query = src_desc.to_format_any();
  weights_desc_query = weights_desc.to_format_any();
  bias_desc_query =
      with_bias ? bias_desc.to_format_any() : ideep::tensor::desc();
  dst_desc_query = dst_desc.to_format_any();

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
      dst_desc_query,
      with_bias,
      bias_desc_query,
      strides,
      dilates,
      padding_l,
      padding_r,
      attr,
      omp_get_max_threads());
  return ideep::utils::computation_cache<
      dnnl::convolution_forward::primitive_desc>::
      fetch_or_create(key, [&]() {
        if (with_bias) {
          return dnnl::convolution_forward::primitive_desc(
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
          return dnnl::convolution_forward::primitive_desc(
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
#endif

// Ported from ideep::convolution_forward::expected_weights_desc (the
// template on is_channels_last).
template <bool is_channels_last = false>
ideep::tensor::desc expected_weights_desc(
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
    const ideep::attr_t& attr = ideep::attr_t()) {
  // weights_dims.size() is 3 for conv1d, 4 for conv2d and 5 for conv3d
  auto src_size = weights_dims.size();
  auto grouped = groups > 1;
  auto weights_dims_g = grouped
      ? ideep::utils::group_dims(weights_dims, groups)
      : weights_dims;
  auto weights_desc = ideep::tensor::desc(weights_dims_g, dtype);

  auto dims_in = weights_desc.get_dims();
  auto ndims = dims_in.size();
  auto dilates_ = ideep::utils::get_compatible_dilates(dilates, src_size);

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      !(aalgorithm == dnnl::algorithm::convolution_winograd &&
        src_dims.empty())
          ? dnnl_success
          : dnnl_invalid_arguments,
      "Incorrect src_dims");
#endif
  dnnl::memory::dims x_dims, y_dims, kernel_size;
  auto ic = groups * dims_in[1 + grouped];
  auto oc = groups * dims_in[0 + grouped];
  if (5 == src_size) {
    kernel_size.push_back(dims_in[ndims - 3]);
    kernel_size.push_back(dims_in[ndims - 2]);
  } else if (4 == src_size) {
    kernel_size.push_back(dims_in[ndims - 2]);
  }
  kernel_size.push_back(dims_in[ndims - 1]);
  if (src_dims.empty()) {
    // Construct a dummy case. Shape from resnet50 model.
    x_dims.push_back(1);
    x_dims.push_back(ic);
    y_dims.push_back(1);
    y_dims.push_back(oc);
    auto valid_x_dim = [=](int idx, int64_t scale) {
      return std::max(
          strides[idx] + ((kernel_size[idx] - 1) * (dilates_[idx] + 1) + 1) -
              (padding_l[idx] + padding_r[idx]),
          scale * kernel_size[idx]);
    };
    x_dims.push_back(valid_x_dim(0, 4));
    if (4 == src_size) {
      x_dims.push_back(valid_x_dim(1, 8));
    } else if (5 == src_size) {
      x_dims.push_back(valid_x_dim(1, 8));
      x_dims.push_back(valid_x_dim(2, 8));
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
    auto out_size = (x_dims[d] -
                     ((kernel_size[d - 2] - 1) * (dilates_[d - 2] + 1) + 1) +
                     (padding_l[d - 2] + padding_r[d - 2])) /
            strides[d - 2] +
        1;
    y_dims.push_back(out_size);
  }
  x_dtype = (dtype == dnnl::memory::data_type::bf16 ||
             dtype == dnnl::memory::data_type::f16)
      ? dtype
      : x_dtype;
  auto y_dtype = dtype != dnnl::memory::data_type::s8
      ? dtype
      : dnnl::memory::data_type::s32;
  ideep::tensor::desc src_desc(x_dims, x_dtype);
  ideep::tensor::desc dst_desc(y_dims, y_dtype);

  auto src_query = src_desc;
  auto dst_query = dst_desc;
  if (is_channels_last) {
    if (4 == src_size) {
      src_query = src_desc.to_format(dnnl::memory::format_tag::nhwc);
      dst_query = dst_desc.to_format(dnnl::memory::format_tag::nhwc);
    } else if (5 == src_size) {
      src_query = src_desc.to_format(dnnl::memory::format_tag::ndhwc);
      dst_query = dst_desc.to_format(dnnl::memory::format_tag::ndhwc);
    } else if (3 == src_size) {
      src_query = src_desc.to_format(dnnl::memory::format_tag::nwc);
      dst_query = dst_desc.to_format(dnnl::memory::format_tag::nwc);
    }
  }

  // FIXME: workaround winograd format issue in inference
  // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is
  // required by winograd primitive. Then, in the cases of variable input
  // shape, the details of dnnl_wino_fmt will be changed. And, extra weights
  // reorder is inevitable each time, leading to bad performance. Here, we set
  // the prop_kind to forward, in order to reorder and cache weights as
  // blocked format, instead of dnnl_wino_fmt.
  auto apkind = aprop_kind;
  if (aalgorithm == dnnl::algorithm::convolution_winograd &&
      aprop_kind == dnnl::prop_kind::forward_inference) {
    apkind = dnnl::prop_kind::forward;
  }
  ideep::attr_t op_attr = attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = get_primitive_desc</*with_bias=*/false>(
      src_query,
      weights_desc,
      ideep::tensor::desc(),
      dst_query,
      strides,
      dilates_,
      padding_l,
      padding_r,
      is_channels_last,
      op_attr,
      aalgorithm,
      apkind);

  // embed group info into weights_desc
#ifdef __aarch64__
  return ideep::tensor::desc(pd.first.weights_desc(), groups);
#else
  return ideep::tensor::desc(pd.weights_desc(), groups);
#endif
}

// Ported from ideep::convolution_forward::do_compute.
// do_compute with given primitive/pd, under the precondition
// that whether or not src/weight/bias/dst need to be reorder
// For both fp32 and int8
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute(
    const convolution_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto scratchpad = ideep::tensor(param.pd.scratchpad_desc());

  auto& expected_src =
      reorder_src ? src.reorder_if_differ_in(param.pd.src_desc()) : src;
  auto&& grouped_weights = weights.make_grouped_weights(param.groups);
  auto&& expected_weights = reorder_weight
      ? grouped_weights.reorder_if_differ_in(param.pd.weights_desc())
      : grouped_weights;
  auto& primitive = param.primitive;
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, bias});
  }
  // Insert tensors of scales and zero points to args
  if (param.all_scales && !param.all_scales->empty()) {
    for (auto& arg_scale_pair : *param.all_scales) {
      int dnnl_arg = arg_scale_pair.first;
      ideep::tensor& scales_m = std::get<1>(arg_scale_pair);
      args.insert({DNNL_ARG_ATTR_SCALES | dnnl_arg, scales_m});
    }
  }
  if (param.all_zero_points && !param.all_zero_points->empty()) {
    for (auto& arg_zp_pair : *param.all_zero_points) {
      int dnnl_arg = arg_zp_pair.first;
      ideep::tensor& zp_m = std::get<1>(arg_zp_pair);
      args.insert({DNNL_ARG_ATTR_ZERO_POINTS | dnnl_arg, zp_m});
    }
  }

  // Deal with dst tensor
  if (reorder_src) {
    auto expected_dst_desc = param.pd.dst_desc();
    ideep::tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != expected_dst_desc) {
      // If dst buffer is not given by user or the given buffer is not in expected format,
      // We need to init a new one
      expected_dst.init(expected_dst_desc);
      if (!dst.is_empty() &&
          param.op_attr.has_op_kind(dnnl::primitive::kind::sum)) {
        // We need to copy the content of given buffer if the op is fused with sum
        expected_dst.feed_from(dst);
      }
    } else {
      // The format of given dst buffer is expected
      expected_dst = dst;
    }
    args.insert({DNNL_ARG_DST, expected_dst});
    primitive.execute(ideep::stream::default_stream(), args);
    // reorder back to dst's buffer if needed
    if (dst.is_empty() || dst.get_desc() == expected_dst.get_desc() ||
        !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
      dst = expected_dst;
    } else {
      dst.feed_from(expected_dst);
    }
  } else {
    args.insert({DNNL_ARG_DST, dst});
    primitive.execute(ideep::stream::default_stream(), args);
  }
}

// Ported from ideep::convolution_forward::do_compute_binary.
// For binary post-op
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute_binary(
    const convolution_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto& pd = param.pd;
  auto& primitive = param.primitive;
  auto scratchpad = ideep::tensor(pd.scratchpad_desc());
  auto& expected_src =
      reorder_src ? src.reorder_if_differ_in(pd.src_desc()) : src;
  if (reorder_src) {
    dst.reinit_if_possible(pd.dst_desc());
  }
  // make sure other has same format with dst.
  bool is_nhwc = dst.get_desc().is_nhwc();
  bool is_ndhwc = dst.get_desc().is_ndhwc();
  auto format_tag = is_nhwc
      ? dnnl::memory::format_tag::nhwc
      : (is_ndhwc ? dnnl::memory::format_tag::ndhwc
                  : dnnl::memory::format_tag::any);
  auto other_desc = other.get_desc().to_format(format_tag);
  auto& expected_other =
      reorder_src ? other.reorder_if_differ_in(other_desc) : other;
  auto&& grouped_weights = weights.make_grouped_weights(param.groups);
  auto&& expected_weights = reorder_weight
      ? grouped_weights.reorder_if_differ_in(pd.weights_desc())
      : grouped_weights;

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  args.insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, expected_other});
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
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, bias});
  }
  primitive.execute(ideep::stream::default_stream(), args);
}

// Ported from ideep::convolution_forward::compute_dispatch.
// For fp32
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_dispatch(
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
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward) {
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
      false,
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

  // Used for to_mkldnn() path
  auto pd = get_primitive_desc<with_bias>(
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

#ifdef __aarch64__
  convolution_forward_params params(
      std::move(pd.first), std::move(pd.second), std::move(op_attr), groups);
#else
  dnnl::convolution_forward primitive(pd);
  convolution_forward_params params(
      std::move(pd), std::move(primitive), std::move(op_attr), groups);
#endif
  do_compute<with_bias, reorder_src, reorder_weight>(
      params, src, weights, bias, dst);
}

// Ported from ideep::convolution_forward::compute_dispatch.
// For int8
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_dispatch(
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
    bool is_channels_last,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::algorithm aalgorithm = dnnl::algorithm::convolution_direct,
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward,
    const ideep::lowp_kind alowp_kind = ideep::u8s8) {
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
      src_scales,
      weights_scales,
      dst_scales,
      src_zero_point,
      dst_zero_point,
      attr,
      alowp_kind,
      with_bias,
      false,
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

  // Used for to_mkldnn() path
  auto pd = get_primitive_desc<with_bias>(
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

#ifdef __aarch64__
  convolution_forward_params param(
      std::move(pd.first),
      std::move(pd.second),
      std::move(op_attr),
      groups,
      std::move(bias_attr));
#else
  dnnl::convolution_forward primitive(pd);
  convolution_forward_params param(
      std::move(pd),
      std::move(primitive),
      std::move(op_attr),
      groups,
      std::move(bias_attr));
#endif

  // Prepare tensors for scales and zero points
  if (param.op_attr.has_scales()) {
    if (!param.all_scales) {
      param.all_scales.reset(new std::unordered_map<int, ideep::tensor>);
    }
    for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
      int dnnl_arg = arg_scale_pair.first;
      const std::vector<float>& scale = arg_scale_pair.second.first;
      ideep::tensor scales_m = ideep::tensor(scale);
      param.all_scales->insert({dnnl_arg, scales_m});
    }
  }
  if (param.op_attr.has_zero_points()) {
    if (!param.all_zero_points) {
      param.all_zero_points.reset(new std::unordered_map<int, ideep::tensor>);
    }
    for (auto& arg_zp_pair : param.op_attr.get_all_zero_points()) {
      int dnnl_arg = arg_zp_pair.first;
      const std::vector<int32_t>& zp = arg_zp_pair.second.first;
      ideep::tensor zp_m = ideep::tensor(zp);
      param.all_zero_points->insert({dnnl_arg, zp_m});
    }
  }
  do_compute<with_bias, reorder_src, reorder_weight>(
      param, src, weights, bias, dst);
}

// Ported from ideep::convolution_forward::do_prepare.
// For fp32
template <bool with_bias>
void do_prepare(
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
      false,
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

  auto pd = get_primitive_desc<with_bias>(
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

#ifdef __aarch64__
  param = {
      std::move(pd.first), std::move(pd.second), std::move(op_attr), groups};
#else
  dnnl::convolution_forward primitive(pd);

  param = {std::move(pd), std::move(primitive), std::move(op_attr), groups};
#endif
}

// Ported from ideep::convolution_forward::compute_binary_dispatch.
// For fp32 with binary post-op
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_binary_dispatch(
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
    dnnl::prop_kind aprop_kind = dnnl::prop_kind::forward) {
  convolution_forward_params params;
  do_prepare<with_bias>(
      params,
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
  do_compute_binary<with_bias, reorder_src, reorder_weight>(
      params, src, other, weights, bias, dst);
}

} // namespace

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
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const ideep::attr_t& attr) {
  bool is_channels_last = src.get_desc().is_channels_last() ||
      weights.get_desc().is_channels_last();
  if (bias.is_empty()) {
    compute_dispatch</*with_bias=*/false, true, true>(
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
        src_scales,
        weights_scales,
        dst_scales,
        src_zero_point,
        dst_zero_point,
        is_channels_last,
        attr);
  } else {
    compute_dispatch</*with_bias=*/true, true, true>(
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
        src_scales,
        weights_scales,
        dst_scales,
        src_zero_point,
        dst_zero_point,
        is_channels_last,
        attr);
  }
}

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
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_point,
    const std::vector<int32_t>& dst_zero_point,
    const ideep::attr_t& attr) {
  bool is_channels_last = src.get_desc().is_channels_last() ||
      weights.get_desc().is_channels_last();
  static ideep::tensor dummy_bias;
  compute_dispatch</*with_bias=*/false, true, true>(
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
      src_scales,
      weights_scales,
      dst_scales,
      src_zero_point,
      dst_zero_point,
      is_channels_last,
      attr);
}

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
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  if (bias.is_empty()) {
    compute_dispatch<false, true, true>(
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
    compute_dispatch<true, true, true>(
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
    bool is_channels_last,
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  static ideep::tensor dummy_bias;
  compute_dispatch<false, true, true>(
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

template <bool reorder_src, bool reorder_weight>
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
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  if (bias.is_empty()) {
    compute_binary_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
        src,
        other,
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
    compute_binary_dispatch</*with_bias=*/true, reorder_src, reorder_weight>(
        src,
        other,
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
    const ideep::attr_t& attr,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind) {
  static ideep::tensor dummy_bias;
  compute_binary_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
      src,
      other,
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

ideep::tensor::desc convolution_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    dnnl::memory::data_type dtype,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const dnnl::memory::dims& dilates,
    int groups,
    dnnl::algorithm aalgorithm,
    dnnl::prop_kind aprop_kind,
    dnnl::memory::data_type x_dtype,
    const dnnl::memory::dims& src_dims,
    const ideep::attr_t& attr,
    bool is_channels_last) {
  if (is_channels_last) {
    return expected_weights_desc<true>(
        weights_dims,
        dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        aprop_kind,
        x_dtype,
        src_dims,
        attr);
  } else {
    return expected_weights_desc<false>(
        weights_dims,
        dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        aprop_kind,
        x_dtype,
        src_dims,
        attr);
  }
}

// The template arguments used by ATen callers.
template void convolution_forward_compute_binary<true, true>(
    const ideep::tensor&,
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
template void convolution_forward_compute_binary<true, true>(
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

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
