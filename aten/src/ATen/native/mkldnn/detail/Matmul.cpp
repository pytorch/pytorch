#include <ATen/native/mkldnn/detail/Matmul.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <memory>
#include <unordered_map>

namespace at::native::mkldnn {

namespace {

// Ported from ideep::matmul_forward_params.
// Common parameters for computation
struct matmul_forward_params {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul primitive;
  ideep::attr_t op_attr;
  ideep::attr_t src_attr;
  ideep::attr_t weights_attr;
  // Keep {dnnl_arg, tensor} pairs of scales for primitive execution
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_scales = nullptr;
};

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
    const ideep::attr_t& attr = ideep::attr_t()) {
#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      src.ndims() == weights.ndims() ? dnnl_success : dnnl_invalid_arguments,
      "Invalid dims in src or weights");
#endif

  ideep::tensor::desc src_desc, weights_desc, bias_desc;
  ideep::attr_t& op_attr = param.op_attr;
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
  // Shares ideep's primitive cache with ideep's own matmul_forward.
  auto key = ideep::utils::create_key(
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc,
      op_attr,
      with_bias,
      omp_get_max_threads());
  param.pd =
      ideep::utils::computation_cache<dnnl::matmul::primitive_desc>::
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

// Ported from ideep::matmul_forward::do_compute (fp32 path).
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

// Ported from ideep::matmul_forward::compute_impl (fp32 branch only).
// For 2-in-1 compute: prepare + compute
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f,
    const ideep::attr_t& attr = ideep::attr_t()) {
  matmul_forward_params param;
  do_prepare<with_bias>(
      param, src, weights, bias, dst, dst_coeff, sum_coeff, attr);
  do_compute<with_bias, reorder_src, reorder_weight>(
      param, src, weights, bias, dst);
}

} // namespace

void matmul_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const float dst_coeff,
    const float sum_coeff,
    const ideep::attr_t& attr) {
  // Consider fp32 only for IPEX
  static ideep::tensor dummy_bias;
  compute_impl</*with_bias=*/false, /*reorder_src*/ true, /*reorder_weight*/ true>(
      src, weights, dummy_bias, dst, dst_coeff, sum_coeff, attr);
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
