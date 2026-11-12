#include <ATen/native/mkldnn/detail/Matmul.h>
#include <ATen/native/mkldnn/detail/PrimitiveCache.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace at::native::mkldnn {

namespace {

// Ported from ideep::matmul_forward::do_prepare.
// For fp32 op
template <bool with_bias>
void do_prepare(
    matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f, // for post-op sum
    const Attr& attr = Attr(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::f32) {
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      src.ndims() == weights.ndims() ? dnnl_success : dnnl_invalid_arguments,
      "Invalid dims in src or weights");
#endif

  ideep::tensor::desc src_desc, weights_desc, bias_desc;
  Attr& op_attr = param.op_attr;
  op_attr = attr;
  auto dst_data_type = dnnl::memory::data_type::f32;
  auto wei_data_type = dnnl::memory::data_type::f32;

  dnnl::memory::dims src_dims = src.get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0]};
  auto ndims = weights.ndims();
  for (auto i = 1; i < ndims - 1; i++) {
    dst_dims.push_back(src_dims[i]);
  }
  dst_dims.push_back(weights.get_dim(ndims - 1));

  // We intentionally didn't set weight desc to format `any` so DNNL wouldn't
  // have to determine weight format for us. Because the weight tensor from
  // pytorch may have a transposed format (say `ba`). However, DNNL would
  // choose plain format for it by default (`ab` in this case), which would
  // introduces *an extra reorder* afterwards. Here we keep the weight format
  // untouched thanks to optimizations for both plain and transposed formats
  // in DNNL.
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      one_of(
          weights.get_data_type(),
          dnnl::memory::data_type::f32,
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)
          ? dnnl_success
          : dnnl_invalid_arguments,
      "Incorrect data type in weights");
#endif

  dst_data_type = src.get_data_type() == dnnl::memory::data_type::bf16
      ? ((dst.get_data_type() == dnnl::memory::data_type::f32)
             ? dnnl::memory::data_type::f32
             : dnnl::memory::data_type::bf16)
      : ((src.get_data_type() == dnnl::memory::data_type::f16)
             ? dnnl::memory::data_type::f16
             : dnnl::memory::data_type::f32);
  src_desc = (src.get_data_type() == dnnl::memory::data_type::bf16 &&
              dst_data_type == dnnl::memory::data_type::f32)
      ? src.get_desc()
      : src.get_desc().to_type(dst_data_type);
  wei_data_type = (src.get_data_type() == dnnl::memory::data_type::bf16 &&
                   dst_data_type == dnnl::memory::data_type::f32)
      ? dnnl::memory::data_type::bf16
      : dst_data_type;
#ifdef __aarch64__
  // for aarch64 ACL backend with fixed format kernels, the weights are
  // always in blocked layout, so, set the descriptor to tag::any for the backend
  // to decide the format
  weights_desc = ideep::tensor::desc(
      weights.get_dims(), wei_data_type, dnnl::memory::format_tag::any);
#else
  // For fp32 matmul, weight (2nd input) is usually not in blocked layout
  // Plain layout runs faster as of oneDNN 3.0
  // Should use tag::any to query blocked layout if there is perf gain later
  weights_desc = weights.get_desc().is_plain()
      ? weights.get_desc().to_type(wei_data_type)
      : ideep::tensor::desc(
            weights.get_dims(), wei_data_type, dnnl::memory::format_tag::any);
#endif

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
    bias_desc = bias.get_desc().to_format_any();
  }

  if (attr.has_op_kind(dnnl::primitive::kind::sum) &&
      attr.get_post_ops().len() == 1) {
    // Extend using fuse_sum to avoid re-init op_attr, and remain the logic
    // of fuse_sum. When re-init op_attr, the attrs in the original attr will
    // be removed.
    dnnl::post_ops po;
    po.append_sum(sum_coeff, 0);
    op_attr.set_post_ops(po);
  }
  if (dst_coeff != 1.0f) {
    // Since onednn 3.0, dst scales are applied after all post ops
    // But here the dst_coeff should be applied before post ops
    // So we set it as src scales here. This keeps the semantics.
    int scale_size = 1;
    op_attr.set_scales(
        DNNL_ARG_SRC,
        ideep::utils::op_scale_mask(scale_size),
        std::vector<float>(1, dst_coeff));
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dst_data_type =
      dst_type == dnnl::memory::data_type::undef ? dst_data_type : dst_type;
  /* rely on tensor descriptor class to se the format appropriately
   * based on the tensor dimensions.
   */
  ideep::tensor::desc dst_desc = ideep::tensor::desc(dst_dims, dst_data_type);
  if (!dst.is_empty()) {
    dst_desc = dst.get_desc().to_type(dst_data_type);
  }

  const auto& engine = ideep::engine::cpu_engine();
#ifdef __aarch64__
  if (with_bias) {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
  } else {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights_desc, dst_desc, op_attr);
  }
  param.primitive = dnnl::matmul(param.pd);
#else
  auto key = utils::create_key(
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc,
      op_attr,
      with_bias,
      omp_get_max_threads());
  param.pd =
      utils::computation_cache<dnnl::matmul::primitive_desc>::
          fetch_or_create(key, [&]() {
            if (with_bias) {
              return dnnl::matmul::primitive_desc(
                  engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
            } else {
              return dnnl::matmul::primitive_desc(
                  engine, src_desc, weights_desc, dst_desc, op_attr);
            }
          });
  param.primitive = dnnl::matmul(param.pd);
#endif
  if (param.op_attr.has_scales()) {
    if (!param.all_scales) {
      param.all_scales.reset(new std::unordered_map<int, ideep::tensor>);
    }
    for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
      int dnnl_arg = arg_scale_pair.first;
      const std::vector<float>& scale = arg_scale_pair.second.first;
      ideep::tensor scales_m(scale);
      param.all_scales->insert({dnnl_arg, scales_m});
    }
  }
}

// Ported from ideep::matmul_forward::do_prepare_static_quant.
// For static int8 op (int8 * int8 -> int8)
template <bool with_bias>
void do_prepare_static_quant(
    matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f, // for post-op sum
    const Attr& attr = Attr(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::u8,
    const ideep::lowp_kind alowp_kind = ideep::u8s8) {
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      src.ndims() == weights.ndims() ? dnnl_success : dnnl_invalid_arguments,
      "Invalid dims in src or weights");
#endif

  ideep::tensor::desc src_desc, weights_desc, bias_desc;
  Attr& op_attr = param.op_attr;
  Attr& src_attr = param.src_attr;
  Attr& weights_attr = param.weights_attr;
  op_attr.set_post_ops(attr.get_post_ops());
  std::vector<float> dst_scales_in;
  auto dst_data_type = dnnl::memory::data_type::u8;

  dnnl::memory::dims src_dims = src.get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], weights.get_dim(1)};
  auto ndims = weights.ndims();
  if (ndims == 3) {
    dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};
  }

  auto& weights_scales_in =
      weights_scales.empty() ? ideep::IDEEP_DEF_SCALE : weights_scales;

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      alowp_kind == ideep::u8s8 || alowp_kind == ideep::s8s8
          ? dnnl_success
          : dnnl_invalid_arguments,
      "Unsupported lowp kind");
#endif

  auto src_scales_in =
      src_scales.empty() ? ideep::IDEEP_DEF_SCALE : src_scales;
  auto src_data_type = (alowp_kind == ideep::u8s8)
      ? dnnl::memory::data_type::u8
      : dnnl::memory::data_type::s8;
  std::vector<int64_t> src_strides = (ndims == 3)
      ? std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1})
      : std::vector<int64_t>({src_dims[1], 1});
  src_desc = ideep::tensor::desc(
      src_dims, src_data_type, dnnl::memory::format_tag::any);
  for (auto& s : src_scales_in) {
    s = 1.0 / s;
  }
  if (src.get_data_type() == dnnl::memory::data_type::f32) {
    src_attr.set_scales(DNNL_ARG_DST, /* mask */ 0, src_scales_in);
  }

  int scale_size = (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
  weights_desc = weights.get_desc();
  if (weights.get_data_type() == dnnl::memory::data_type::f32) {
    weights_attr = {
        ideep::utils::tensor_scale_mask(scale_size, false), weights_scales_in};
  }

  // determine dst data type
  if (dst.get_data_type() != dnnl::memory::data_type::undef) {
    dst_data_type = dst.get_data_type();
  } else if (dst_scales.empty() || dst_scales == ideep::IDEEP_DEF_SCALE) {
    dst_data_type = dnnl::memory::data_type::f32;
  } else {
    dst_data_type = dnnl::memory::data_type::u8;
  }

  // fill primitive attr
  dst_scales_in =
      (dst_scales.empty() || dst_data_type == dnnl::memory::data_type::f32)
      ? ideep::IDEEP_DEF_SCALE
      : dst_scales;
  const auto& src_zero_point = src.has_zero_point() ? src.get_zero_point()
      : src_zero_points.empty()                     ? ideep::IDEEP_DEF_ZP
                                                    : src_zero_points;
  const auto& dst_zero_point = dst.has_zero_point() ? dst.get_zero_point()
      : dst_zero_points.empty()                     ? ideep::IDEEP_DEF_ZP
                                                    : dst_zero_points;
#ifndef NDEBUG
  const auto src_zero_point_size =
      static_cast<dnnl::memory::dim>(src_zero_point.size());
  const auto dst_zero_point_size =
      static_cast<dnnl::memory::dim>(dst_zero_point.size());
  dnnl::error::wrap_c_api(
      src_zero_point_size == 1 && dst_zero_point_size == 1
          ? dnnl_success
          : dnnl_invalid_arguments,
      "DNNL only support 1-dim zero_point");
#endif

  if (attr.has_op_kind(dnnl::primitive::kind::sum)) {
    float sum_scale =
        sum_coeff / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
    op_attr = Attr::fuse_sum(sum_scale);
  }

  if (src_scales_in[0] != 1.0f) {
    op_attr.set_scales(
        DNNL_ARG_SRC,
        ideep::utils::op_scale_mask(src_scales_in.size()),
        src_scales_in);
  }
  auto wei_scales = weights_scales_in;
  if (!std::all_of(wei_scales.begin(), wei_scales.end(), [](float i) {
        return i == 1.0f;
      })) {
    for (auto& s : wei_scales) {
      s = 1.0 / s;
    }
    op_attr.set_scales(
        DNNL_ARG_WEIGHTS,
        ideep::utils::op_scale_mask(wei_scales.size()),
        wei_scales);
  }
  for (auto& s : dst_scales_in) {
    s = dst_coeff / s;
  }
  if (dst_scales_in[0] != 1.0f) {
    op_attr.set_scales(
        DNNL_ARG_DST,
        ideep::utils::op_scale_mask(dst_scales_in.size()),
        dst_scales_in);
  }
  if (src_zero_point[0] != 0) {
    op_attr.set_zero_points(
        DNNL_ARG_SRC,
        ideep::utils::tensor_zp_mask(src_zero_point.size()),
        src_zero_point);
  }
  if (src.get_data_type() == dnnl::memory::data_type::f32) {
    // Set zero point for src reorder (fp32 -> int8).
    // First arg should be DNNL_ARG_DST rather than DNNL_ARG_SRC
    src_attr.set_zero_points(
        DNNL_ARG_DST,
        ideep::utils::tensor_zp_mask(src_zero_point.size()),
        src_zero_point);
  }
  if (dst_data_type != dnnl::memory::data_type::f32 &&
      dst_zero_point[0] != 0) {
    op_attr.set_zero_points(
        DNNL_ARG_DST,
        ideep::utils::tensor_zp_mask(dst_zero_point.size()),
        dst_zero_point);
  }

  if (with_bias) {
    dnnl::memory::format_tag bia_tag = bias.get_dims().size() == 2
        ? dnnl::memory::format_tag::ab
        : dnnl::memory::format_tag::abc;
    bias_desc = {
        bias.get_dims(),
        dnnl::memory::data_type::f32,
        bia_tag}; // Use f32 instead of s32 to improve accuracy
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dst_data_type =
      dst_type == dnnl::memory::data_type::undef ? dst_data_type : dst_type;
  ideep::tensor::desc dst_desc = ideep::tensor::desc(
      dst_dims, dst_data_type, dnnl::memory::format_tag::any);
  if (!dst.is_empty()) {
    dst_desc = dst.get_desc().to_type(dst_data_type);
  }
  const auto& engine = ideep::engine::cpu_engine();
#ifdef __aarch64__
  if (with_bias) {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
  } else {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights_desc, dst_desc, op_attr);
  }
  param.primitive = dnnl::matmul(param.pd);
#else
  auto key = utils::create_key(
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc,
      op_attr,
      with_bias,
      omp_get_max_threads());
  param.pd =
      utils::computation_cache<dnnl::matmul::primitive_desc>::
          fetch_or_create(key, [&]() {
            if (with_bias) {
              return dnnl::matmul::primitive_desc(
                  engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
            } else {
              return dnnl::matmul::primitive_desc(
                  engine, src_desc, weights_desc, dst_desc, op_attr);
            }
          });
  param.primitive = dnnl::matmul(param.pd);
#endif
  if (param.op_attr.has_scales()) {
    if (!param.all_scales) {
      param.all_scales.reset(new std::unordered_map<int, ideep::tensor>);
    }
    for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
      int dnnl_arg = arg_scale_pair.first;
      const std::vector<float>& scale = arg_scale_pair.second.first;
      ideep::tensor scales_m(scale);
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
      ideep::tensor zp_m(zp);
      param.all_zero_points->insert({dnnl_arg, zp_m});
    }
  }
}

// Ported from ideep::matmul_forward::do_prepare_dynamic_quant.
// For dynamic int8 op (fp32 * int8 -> fp32)
template <bool with_bias>
void do_prepare_dynamic_quant(
    matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& weights_scales,
    const float sum_coeff = 1.0f, // for post-op sum
    const Attr& attr = Attr(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::f32,
    const ideep::lowp_kind alowp_kind = ideep::u8s8) {
  /* This function does the following things:
   * - Determine expected descs of src/weight/dst
   * - Use runtime values for op attributes
   * - Create matmul primitive desc and primitive
   * - Create reorder primitive for src (fp32 -> int8)
   */

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      src.ndims() == weights.ndims() ? dnnl_success : dnnl_invalid_arguments,
      "Invalid dims in src or weights");
#endif
  if (!param.dq_param_ptr) {
    param.dq_param_ptr = std::make_shared<matmul_forward_dyn_quant_params>();
  }
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      param.dq_param_ptr ? dnnl_success : dnnl_invalid_arguments,
      "Failed to allocate memory for parameters");
#endif

  ideep::tensor::desc& src_desc = param.dq_param_ptr->src_desc;
  Attr& op_attr = param.op_attr;
  Attr src_attr;

  dnnl::memory::dims src_dims = src.get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], weights.get_dim(1)};
  auto ndims = weights.ndims();
  if (ndims == 3)
    dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};

  auto& weights_scales_in =
      weights.has_scale() ? weights.get_scale() : weights_scales;

  auto src_data_type = alowp_kind == ideep::s8s8 ? dnnl::memory::data_type::s8
                                                 : dnnl::memory::data_type::u8;
  std::vector<int64_t> src_strides = (ndims == 3)
      ? std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1})
      : std::vector<int64_t>({src_dims[1], 1});
  src_desc = ideep::tensor::desc(src_dims, src_data_type, src_strides);

  // Post-ops
  auto pops = attr.get_post_ops();
  dnnl::post_ops new_pops;
  for (int i = 0; i < pops.len(); ++i) {
    // Only sum and eltwise is supported now
    if (dnnl::primitive::kind::sum == pops.kind(i)) {
      // The parameter sum_coeff is passed in explicitly now due to legacy code.
      // TO-DO:
      // Remove the argument 'sum_coeff'.
      // User should prepare all post-ops in argument 'attr'.
      new_pops.append_sum(sum_coeff);
    } else if (dnnl::primitive::kind::eltwise == pops.kind(i)) {
      float alpha = 1.0, beta = 0.0;
      dnnl::algorithm alg;
      pops.get_params_eltwise(i, alg, alpha, beta);
      new_pops.append_eltwise(alg, alpha, beta);
    }
  }
  op_attr.set_post_ops(new_pops);

  // Scales and zero points of src are obtained at runtime. So only set mask here
  // Weight zero point is always 0. Do not set.
  // Dst is in fp32, no scales or zero points.
  op_attr.set_scales_mask(
      DNNL_ARG_SRC, ideep::utils::op_scale_mask(1 /* scale_size */));
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, ideep::utils::tensor_zp_mask(1));
  auto wei_scales = weights_scales_in;
  for (auto& s : wei_scales) {
    s = 1.0 / s;
  }
  op_attr.set_scales(
      DNNL_ARG_WEIGHTS,
      ideep::utils::op_scale_mask(wei_scales.size()),
      wei_scales);
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Src attr for reorder
  src_attr.set_scales_mask(DNNL_ARG_DST, ideep::utils::op_scale_mask(1));
  src_attr.set_zero_points_mask(DNNL_ARG_DST, ideep::utils::tensor_zp_mask(1));

  // Dst desc
  std::vector<int64_t> dst_strides = (ndims == 3)
      ? std::vector<int64_t>({dst_dims[2] * dst_dims[1], dst_dims[1], 1})
      : std::vector<int64_t>({dst_dims[1], 1});
  ideep::tensor::desc dst_desc =
      ideep::tensor::desc(dst_dims, dst_type, dst_strides);
  if (!dst.is_empty()) {
    dst_desc = dst.get_desc().to_type(dst_type);
  }
  ideep::tensor::desc bias_desc;
  if (with_bias) {
    dnnl::memory::format_tag bia_tag = bias.get_dims().size() == 2
        ? dnnl::memory::format_tag::ab
        : dnnl::memory::format_tag::abc;
    bias_desc = {bias.get_dims(), dnnl::memory::data_type::f32, bia_tag};
  }

  // Create pd and primitive
  const auto& engine = ideep::engine::cpu_engine();
#ifdef __aarch64__
  if (with_bias) {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights.get_desc(), bias_desc, dst_desc, op_attr);
  } else {
    param.pd = dnnl::matmul::primitive_desc(
        engine, src_desc, weights.get_desc(), dst_desc, op_attr);
  }
  param.primitive = dnnl::matmul(param.pd);
#else
  auto key = utils::create_key(
      src_desc,
      weights.get_desc(),
      bias_desc,
      dst_desc,
      op_attr,
      with_bias,
      omp_get_max_threads());
  param.pd =
      utils::computation_cache<dnnl::matmul::primitive_desc>::
          fetch_or_create(key, [&]() {
            if (with_bias) {
              return dnnl::matmul::primitive_desc(
                  engine,
                  src_desc,
                  weights.get_desc(),
                  bias_desc,
                  dst_desc,
                  op_attr);
            } else {
              return dnnl::matmul::primitive_desc(
                  engine, src_desc, weights.get_desc(), dst_desc, op_attr);
            }
          });
  param.primitive = dnnl::matmul(param.pd);
#endif

  // Create src reorder primitive with runtime scales/zero point
  auto src_reorder_pd = dnnl::reorder::primitive_desc(
      engine, src.get_desc(), engine, src_desc, src_attr);
  param.dq_param_ptr->src_reorder = dnnl::reorder(src_reorder_pd);

  if (param.op_attr.has_scales()) {
    if (!param.all_scales) {
      param.all_scales.reset(new std::unordered_map<int, ideep::tensor>);
    }
    for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
      int dnnl_arg = arg_scale_pair.first;
      const std::vector<float>& scale = arg_scale_pair.second.first;
      ideep::tensor scales_m(scale);
      param.all_scales->insert({dnnl_arg, scales_m});
    }
  }
}

// Ported from ideep::matmul_forward::do_compute.
// For fp32 and static int8 op (int8 * int8 -> int8)
// Set reorder flags to false if you are sure the memory layout aligns
// with primitive descriptor. Otherwise, checks are made and reorder
// may be needed.
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto& pd = param.pd;
  auto& primitive = param.primitive;
  auto& op_attr = param.op_attr;
  auto& src_attr = param.src_attr;
  auto& weights_attr = param.weights_attr;

  auto expected_src_desc = pd.src_desc();
  auto expected_wei_desc = pd.weights_desc();
  auto expected_dst_desc = pd.dst_desc();

  auto& expected_src = reorder_src
      ? src.reorder_if_differ_in(expected_src_desc, src_attr)
      : src;
  auto& expected_weights = reorder_weight
      ? weights.reorder_if_differ_in(expected_wei_desc, weights_attr)
      : weights;
  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, bias});
  }
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
  if (reorder_src) {
    ideep::tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != expected_dst_desc) {
      // If dst buffer are not given by user or user given dst buffer are not
      // under expected format
      // We need init a new one
      expected_dst.init(expected_dst_desc);
      if (!dst.is_empty() && op_attr.has_op_kind(dnnl::primitive::kind::sum)) {
        // We need copy the content of given buffer if matmul is fused with sum
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

// Ported from ideep::matmul_forward::do_compute_dynamic_quant.
// For dynamic int8 op (fp32 * int8 -> fp32)
// Set reorder flags to false if you are sure the memory layout aligns
// with primitive descriptor. Otherwise, checks are made and reorder
// may be needed.
template <bool with_bias, bool reorder_weight = true>
void do_compute_dynamic_quant(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<int32_t>& src_zero_points,
    const float dst_coeff = 1.0f) {
  /* Compute for dynamic quantized linear. This function does the following things:
   * - Get matmul primitive from param
   * - Get reorder primitive for src from param.dq_param_ptr
   * - Prepare tensors of output scales and zero points.
   * - Compute by executing matmul primitive
   */

  // Get primitive, etc. from param
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      param.dq_param_ptr ? dnnl_success : dnnl_invalid_arguments,
      "Parameters for dynamic quantization not found");
#endif
  auto& pd = param.pd;
  auto& primitive = param.primitive;
  auto& weights_attr = param.weights_attr;
  auto& expected_src_desc = param.dq_param_ptr->src_desc;
  auto& src_reorder = param.dq_param_ptr->src_reorder;

  // Prepare tensor of output scales
  auto src_scales_in = src.has_scale()
      ? src.get_scale()
      : (src_scales.empty() ? ideep::IDEEP_DEF_SCALE : src_scales);

  const dnnl::memory::dim scale_zp_stride = 1;

  // Prepare tensor of src scales
  for (auto& s : src_scales_in) {
    s = 1.0 / s;
  }
  int src_scale_size = src_scales_in.size();
  ideep::tensor::desc src_scales_desc = {
      {src_scale_size}, dnnl::memory::data_type::f32, {scale_zp_stride}};
  const auto& engine = ideep::engine::cpu_engine();
  ideep::tensor src_scales_m(
      src_scales_desc,
      reinterpret_cast<float*>(src_scales_in.data()),
      engine);

  // Prepare tensor of src zero point
  auto src_zero_point = src.has_zero_point() ? src.get_zero_point()
      : src_zero_points.empty()              ? ideep::IDEEP_DEF_ZP
                                             : src_zero_points;
  const auto src_zero_point_size =
      static_cast<dnnl::memory::dim>(src_zero_point.size());
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      src_zero_point_size == 1 ? dnnl_success : dnnl_invalid_arguments,
      "DNNL only support 1-dim zero_point");
#endif
  ideep::tensor::desc src_zero_point_desc = {
      {src_zero_point_size}, dnnl::memory::data_type::s32, {scale_zp_stride}};
  ideep::tensor src_zero_point_m(
      src_zero_point_desc,
      reinterpret_cast<int32_t*>(src_zero_point.data()),
      engine);

  // Reroder src (f32 -> u8)
  ideep::tensor expected_src(expected_src_desc);
  src_reorder.execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_FROM, src},
       {DNNL_ARG_TO, expected_src},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, src_scales_m},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, src_zero_point_m}});

  // Check weight desc
  auto& expected_weights = reorder_weight
      ? weights.reorder_if_differ_in(pd.weights_desc(), weights_attr)
      : weights;
  ideep::tensor& expected_dst = dst;

  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  args.insert({DNNL_ARG_DST, expected_dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, bias});
  }
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
  // Src scales and zero point are obtained at runtime thus set separately
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_m});
  args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m});

  primitive.execute(ideep::stream::default_stream(), args);
}

// Ported from ideep::matmul_forward::compute_impl.
// For 2-in-1 compute: prepare + compute
// Supports fp32, static int8 and dynamic int8
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales = std::vector<float>(),
    const std::vector<float>& weights_scales = std::vector<float>(),
    const std::vector<float>& dst_scales = std::vector<float>(),
    const std::vector<int32_t>& src_zero_points = std::vector<int32_t>(),
    const std::vector<int32_t>& dst_zero_points = std::vector<int32_t>(),
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f,
    const Attr& attr = Attr(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::undef,
    const ideep::lowp_kind alowp_kind = ideep::u8s8) {
  matmul_forward_params param;
  auto& weights_scales_in =
      weights.has_scale() ? weights.get_scale() : weights_scales;
  if (!weights_scales_in.empty()) { // for int8
    if (src.get_data_type() ==
        dnnl::memory::data_type::f32) { // invoke dynamic quant
      // prepare
      do_prepare_dynamic_quant<with_bias>(
          param,
          src,
          weights,
          bias,
          dst,
          weights_scales,
          sum_coeff,
          attr,
          dnnl::memory::data_type::f32,
          alowp_kind);
      // compute
      if (bias.is_empty()) {
        do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
            param, src, weights, bias, dst, src_scales, src_zero_points, dst_coeff);
      } else {
        do_compute_dynamic_quant</*with_bias=*/true, reorder_weight>(
            param, src, weights, bias, dst, src_scales, src_zero_points, dst_coeff);
      }
      return;
    } else { // invoke static quant
      do_prepare_static_quant<with_bias>(
          param,
          src,
          weights,
          bias,
          dst,
          src_scales,
          weights_scales,
          dst_scales,
          src_zero_points,
          dst_zero_points,
          dst_coeff,
          sum_coeff,
          attr,
          dst_type,
          alowp_kind);
    }
  } else {
    do_prepare<with_bias>(
        param, src, weights, bias, dst, dst_coeff, sum_coeff, attr, dst_type);
  }
  do_compute<with_bias, reorder_src, reorder_weight>(
      param, src, weights, bias, dst);
}

} // namespace

template <bool reorder_src, bool reorder_weight>
void matmul_forward_compute(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const float dst_coeff,
    const float sum_coeff,
    const Attr& attr,
    const dnnl::memory::data_type dst_type,
    const ideep::lowp_kind alowp_kind) {
  if (bias.is_empty()) {
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        src_zero_points,
        dst_zero_points,
        dst_coeff,
        sum_coeff,
        attr,
        dst_type,
        alowp_kind);
  } else {
    compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
        src,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        src_zero_points,
        dst_zero_points,
        dst_coeff,
        sum_coeff,
        attr,
        dst_type,
        alowp_kind);
  }
}

template <bool is_dynamic>
void matmul_forward_prepare(
    matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const float dst_coeff,
    const float sum_coeff,
    const Attr& attr,
    const dnnl::memory::data_type dst_type,
    const ideep::lowp_kind alowp_kind) {
  if (is_dynamic) {
    if (bias.is_empty()) {
      do_prepare_dynamic_quant</*with_bias=*/false>(
          param,
          src,
          weights,
          bias,
          dst,
          weights_scales,
          sum_coeff,
          attr,
          dnnl::memory::data_type::f32,
          alowp_kind);
    } else {
      do_prepare_dynamic_quant</*with_bias=*/true>(
          param,
          src,
          weights,
          bias,
          dst,
          weights_scales,
          sum_coeff,
          attr,
          dnnl::memory::data_type::f32,
          alowp_kind);
    }
  } else {
    if (bias.is_empty()) {
      do_prepare_static_quant</*with_bias=*/false>(
          param,
          src,
          weights,
          bias,
          dst,
          src_scales,
          weights_scales,
          dst_scales,
          src_zero_points,
          dst_zero_points,
          dst_coeff,
          sum_coeff,
          attr,
          dst_type,
          alowp_kind);
    } else {
      do_prepare_static_quant</*with_bias=*/true>(
          param,
          src,
          weights,
          bias,
          dst,
          src_scales,
          weights_scales,
          dst_scales,
          src_zero_points,
          dst_zero_points,
          dst_coeff,
          sum_coeff,
          attr,
          dst_type,
          alowp_kind);
    }
  }
}

template <bool reorder_src, bool reorder_weight>
void matmul_forward_compute(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  if (bias.is_empty()) {
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, bias, dst);
  } else {
    do_compute</*with_bias=*/true, reorder_src, reorder_weight>(
        param, src, weights, bias, dst);
  }
}

template <bool reorder_weight>
void matmul_forward_compute(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<int32_t>& src_zero_points,
    const float dst_coeff) {
  if (bias.is_empty()) {
    do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
        param, src, weights, bias, dst, src_scales, src_zero_points, dst_coeff);
  } else {
    do_compute_dynamic_quant</*with_bias=*/true, reorder_weight>(
        param, src, weights, bias, dst, src_scales, src_zero_points, dst_coeff);
  }
}

void matmul_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const float dst_coeff,
    const float sum_coeff,
    const Attr& attr) {
  // Consider fp32 only for IPEX
  static ideep::tensor dummy_bias;
  static ideep::lowp_kind dummy_lowp_kind = ideep::u8s8;
  compute_impl</*with_bias=*/false, /*reorder_src*/ true, /*reorder_weight*/ true>(
      src,
      weights,
      dummy_bias,
      dst,
      ideep::IDEEP_EMPTY_SCALE,
      ideep::IDEEP_EMPTY_SCALE,
      ideep::IDEEP_EMPTY_SCALE,
      ideep::IDEEP_EMPTY_ZP,
      ideep::IDEEP_EMPTY_ZP,
      dst_coeff,
      sum_coeff,
      attr,
      dnnl::memory::data_type::undef,
      dummy_lowp_kind);
}

ideep::tensor::desc matmul_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims,
    dnnl::memory::data_type dtype,
    dnnl::memory::data_type x_dtype,
    dnnl::memory::data_type y_dtype,
    const Attr& attr) {
  auto ndims = weights_dims.size();
  auto x_dims = weights_dims;
  x_dims[ndims - 2] = src_dims.size() > 0 && src_dims.size() == ndims
      ? src_dims[ndims - 2]
      : 1;
  x_dims[ndims - 1] = weights_dims[ndims - 2];
  dnnl::memory::dims y_dims = (ndims == 3)
      ? dnnl::memory::dims({x_dims[0], x_dims[1], weights_dims[2]})
      : dnnl::memory::dims({x_dims[0], weights_dims[1]});

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      x_dims.size() == weights_dims.size() ? dnnl_success
                                           : dnnl_invalid_arguments,
      "Invalid dims for data and weights");
#endif
  ideep::tensor::desc x_desc(
      x_dims,
      x_dtype,
      ndims == 2 ? dnnl::memory::format_tag::ab
                 : dnnl::memory::format_tag::abc);
  ideep::tensor::desc y_desc(
      y_dims,
      y_dtype,
      ndims == 2 ? dnnl::memory::format_tag::ab
                 : dnnl::memory::format_tag::abc);
  ideep::tensor::desc weights_desc(
      weights_dims, dtype, dnnl::memory::format_tag::any);
  auto pd = dnnl::matmul::primitive_desc(
      ideep::engine::cpu_engine(), x_desc, weights_desc, y_desc, attr);
  return pd.weights_desc();
}

ideep::tensor::desc matmul_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims,
    dnnl::memory::data_type dtype,
    dnnl::memory::data_type x_dtype,
    const Attr& attr) {
  auto y_dtype = (dtype != dnnl::memory::data_type::s8)
      ? dtype
      : dnnl::memory::data_type::s32;

  return matmul_forward_expected_weights_desc(
      weights_dims, src_dims, dtype, x_dtype, y_dtype, attr);
}

template void matmul_forward_compute<true, true>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<int32_t>&,
    const std::vector<int32_t>&,
    const float,
    const float,
    const Attr&,
    const dnnl::memory::data_type,
    const ideep::lowp_kind);

template void matmul_forward_prepare<false>(
    matmul_forward_params&,
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<int32_t>&,
    const std::vector<int32_t>&,
    const float,
    const float,
    const Attr&,
    const dnnl::memory::data_type,
    const ideep::lowp_kind);

template void matmul_forward_prepare<true>(
    matmul_forward_params&,
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<int32_t>&,
    const std::vector<int32_t>&,
    const float,
    const float,
    const Attr&,
    const dnnl::memory::data_type,
    const ideep::lowp_kind);

template void matmul_forward_compute<false, false>(
    const matmul_forward_params&,
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&);

template void matmul_forward_compute<true>(
    const matmul_forward_params&,
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const std::vector<float>&,
    const std::vector<int32_t>&,
    const float);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
