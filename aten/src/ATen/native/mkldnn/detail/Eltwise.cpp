#include <ATen/native/mkldnn/detail/Eltwise.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void eltwise_forward(
    const ideep::tensor& src,
    ideep::tensor& dst,
    dnnl::algorithm algorithm,
    dnnl::prop_kind prop_kind,
    float alpha,
    float beta) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src_in = src;
  // Quantized inputs are dequantized here for every algorithm except relu.
  if (algorithm != dnnl::algorithm::eltwise_relu &&
      one_of(
          src.get_data_type(),
          dnnl::memory::data_type::s8,
          dnnl::memory::data_type::u8)) {
    src_in = src_in.dequantize();
  }
  auto src_desc = src_in.get_desc();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::eltwise_forward::primitive_desc(
      engine, prop_kind, algorithm, src_desc, src_desc, alpha, beta, op_attr);

  // May replace dst's buffer and layout when they differ from pd.dst_desc().
  dst.reinit_if_possible(pd.dst_desc());
  if (src_in.has_scale()) {
    dst.set_scale(src_in.get_scale());
  }
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::eltwise_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC, src_in},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  if (dst.has_scale() && algorithm == dnnl::algorithm::eltwise_relu &&
      dst.get_data_type() == dnnl::memory::data_type::s8) {
    dst.to_type(dnnl::memory::data_type::u8);
  }
}

void eltwise_backward(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_src,
    dnnl::algorithm algorithm,
    float alpha,
    float beta) {
  if (is_avx2_vnni_2() &&
      one_of(
          src.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc = src.get_desc();

  auto forward_hints = dnnl::eltwise_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      algorithm,
      src_desc,
      src_desc,
      alpha,
      beta);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::eltwise_backward::primitive_desc(
      engine,
      algorithm,
      forward_hints.src_desc(),
      forward_hints.dst_desc(),
      src_desc,
      alpha,
      beta,
      forward_hints,
      op_attr);

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  ideep::tensor expected_diff_src;
  if (diff_src.is_empty() || diff_src.get_desc() != pd.diff_src_desc()) {
    expected_diff_src.init(pd.diff_src_desc());
  } else {
    expected_diff_src = diff_src;
  }

  const bool use_dst = one_of(
      algorithm,
      dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
      dnnl::algorithm::eltwise_tanh_use_dst_for_bwd,
      dnnl::algorithm::eltwise_elu_use_dst_for_bwd,
      dnnl::algorithm::eltwise_sqrt_use_dst_for_bwd,
      dnnl::algorithm::eltwise_logistic_use_dst_for_bwd,
      dnnl::algorithm::eltwise_exp_use_dst_for_bwd);
  auto src_dst_arg = use_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;
  auto expected_src_dst_desc = use_dst ? pd.dst_desc() : pd.src_desc();
  auto expected_src_dst = src.reorder_if_differ_in(expected_src_dst_desc);
  ideep::tensor scratchpad(pd.scratchpad_desc());
  dnnl::eltwise_backward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
       {src_dst_arg, expected_src_dst},
       {DNNL_ARG_DIFF_SRC, expected_diff_src},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  if (diff_src != expected_diff_src) {
    // A caller-provided diff_src with the same shape keeps its buffer and
    // layout; otherwise it is replaced.
    if (!diff_src.is_empty() &&
        diff_src.get_desc().has_same_shape_as(expected_diff_src.get_desc())) {
      diff_src.feed_from(expected_diff_src);
    } else {
      diff_src = expected_diff_src;
    }
  }
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
