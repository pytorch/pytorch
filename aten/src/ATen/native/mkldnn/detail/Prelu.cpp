#include <ATen/native/mkldnn/detail/Prelu.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

namespace {

// Weight is 1-D (channels); oneDNN needs it broadcastable to src's rank.
ideep::tensor weight_with_src_rank(
    const ideep::tensor& src,
    const ideep::tensor& weight) {
  auto weight_in = weight;
  if (src.ndims() != weight.ndims()) {
    dnnl::memory::dims dim_w(src.ndims(), 1);
    dim_w[1] = weight.get_dim(0);
    weight_in.reshape(dim_w);
  }
  return weight_in;
}

} // namespace

void prelu_forward(
    const ideep::tensor& src,
    const ideep::tensor& weight,
    ideep::tensor& dst,
    dnnl::prop_kind prop_kind) {
  const auto& engine = ideep::engine::cpu_engine();
  auto weight_in = weight_with_src_rank(src, weight);

  auto src_desc = src.get_desc();
  auto weight_desc = weight_in.get_desc().to_format_any();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::prelu_forward::primitive_desc(
      engine, prop_kind, src_desc, weight_desc, src_desc, op_attr);
  auto expected_weights = weight_in.reorder_if_differ_in(pd.weights_desc());
  // May replace dst's buffer and layout when they differ from pd.dst_desc().
  dst.reinit_if_possible(pd.dst_desc());

  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::prelu_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC, src},
       {DNNL_ARG_WEIGHTS, expected_weights},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

void prelu_backward(
    const ideep::tensor& src,
    const ideep::tensor& weight,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_src,
    ideep::tensor& diff_weight) {
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = ideep::engine::cpu_engine();
  auto weight_dims = weight.get_dims();
  auto weight_in = weight_with_src_rank(src, weight);

  auto src_desc = src.get_desc();
  auto weight_desc = weight_in.get_desc().to_format_any();
  auto diff_dst_desc = diff_dst.get_desc();
  auto diff_weights_desc =
      ideep::tensor::desc(
          weight_in.get_dims(),
          diff_dst.get_data_type(),
          dnnl::memory::format_tag::any)
          .to_format_any();
  auto forward_hints = dnnl::prelu_forward::primitive_desc(
      engine, dnnl::prop_kind::forward, src_desc, weight_desc, src_desc);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::prelu_backward::primitive_desc(
      engine,
      src_desc,
      weight_desc,
      diff_dst_desc,
      diff_weights_desc,
      diff_dst_desc,
      forward_hints,
      op_attr);

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  auto expected_weights = weight_in.reorder_if_differ_in(pd.weights_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());
  diff_weight.reinit_if_possible(pd.diff_weights_desc());

  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::prelu_backward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
       {DNNL_ARG_SRC, expected_src},
       {DNNL_ARG_WEIGHTS, expected_weights},
       {DNNL_ARG_DIFF_SRC, diff_src},
       {DNNL_ARG_DIFF_WEIGHTS, diff_weight},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  if (diff_weight.get_dims() != weight_dims) {
    diff_weight.reshape(weight_dims);
  }
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
