#include <ATen/native/mkldnn/detail/Pooling.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <unordered_map>

namespace at::native::mkldnn {

void pooling_forward(
    const ideep::tensor& src,
    const dnnl::memory::dims& output_sizes,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::algorithm algorithm,
    dnnl::prop_kind prop_kind) {
  const auto& engine = ideep::engine::cpu_engine();
  const bool with_workspace = prop_kind == dnnl::prop_kind::forward_training &&
      algorithm == dnnl::algorithm::pooling_max;

  // Workaround for oneDNN issue intel/mkl-dnn#588 (kept from ideep).
  auto src_desc = src._get_unblocked_desc_if_4c_blocked();

  ideep::tensor::desc dst_desc(
      output_sizes, src.get_data_type(), dnnl::memory::format_tag::any);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  const dnnl::memory::dims dilation(strides.size(), 0);

  auto pd = dnnl::pooling_forward::primitive_desc(
      engine,
      prop_kind,
      algorithm,
      src_desc,
      dst_desc,
      strides,
      kernel,
      dilation,
      padding_l,
      padding_r,
      op_attr);

  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  dst.reinit_if_possible(pd.dst_desc());
  if (src.has_scale()) {
    dst.set_scale(src.get_scale());
  }

  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, expected_src},
      {DNNL_ARG_DST, dst},
      {DNNL_ARG_SCRATCHPAD, scratchpad}};
  if (with_workspace) {
    dst.init_workspace(pd.workspace_desc());
    args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
  }

  dnnl::pooling_forward(pd).execute(ideep::stream::default_stream(), args);
}

void pooling_backward(
    const ideep::tensor& diff_dst,
    const ideep::tensor& dst,
    const ideep::tensor& src,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::algorithm algorithm) {
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
  auto dst_desc = dst.get_desc();

  const dnnl::memory::dims dilation(strides.size(), 0);

  auto forward_hints = dnnl::pooling_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      algorithm,
      src_desc,
      dst_desc,
      strides,
      kernel,
      dilation,
      padding_l,
      padding_r);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::pooling_backward::primitive_desc(
      engine,
      algorithm,
      src_desc,
      dst_desc,
      strides,
      kernel,
      dilation,
      padding_l,
      padding_r,
      forward_hints,
      op_attr);

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());

  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DIFF_DST, expected_diff_dst},
      {DNNL_ARG_DIFF_SRC, diff_src},
      {DNNL_ARG_SCRATCHPAD, scratchpad}};

  if (dst.has_workspace()) {
    auto expected_workspace =
        dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
    args.insert({DNNL_ARG_WORKSPACE, expected_workspace});
  }

  dnnl::pooling_backward(pd).execute(ideep::stream::default_stream(), args);
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
