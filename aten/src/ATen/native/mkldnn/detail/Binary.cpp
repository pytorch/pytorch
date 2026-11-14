#include <ATen/native/mkldnn/detail/Binary.h>

#if AT_MKLDNN_ENABLED()

#include <unordered_map>

namespace at::native::mkldnn {

void sum(
    const std::vector<float>& scales,
    const std::vector<ideep::tensor>& srcs,
    ideep::tensor& dst) {
  const auto& engine = ideep::engine::cpu_engine();
  std::vector<dnnl::memory::desc> src_descs;
  src_descs.reserve(srcs.size());
  for (const auto& src : srcs) {
    src_descs.emplace_back(src.get_desc());
  }

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::sum::primitive_desc(engine, scales, src_descs, op_attr);

  // May replace dst's buffer and layout when they differ from pd.dst_desc().
  dst.reinit_if_possible(pd.dst_desc());
  ideep::tensor scratchpad(pd.scratchpad_desc());
  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratchpad}};
  for (size_t i = 0; i < srcs.size(); ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + static_cast<int>(i), srcs[i]});
  }

  dnnl::sum(pd).execute(ideep::stream::default_stream(), args);
}

void binary(
    const ideep::tensor& src0,
    const ideep::tensor& src1,
    ideep::tensor& dst,
    dnnl::algorithm algorithm) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src0_desc = src0.get_desc();
  auto src1_desc = src1.get_desc();
  auto dst_desc = src0_desc.to_format_any();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::binary::primitive_desc(
      engine, algorithm, src0_desc, src1_desc, dst_desc, op_attr);

  ideep::tensor scratchpad(pd.scratchpad_desc());

  auto expected_src0 = src0.reorder_if_differ_in(pd.src0_desc());
  auto expected_src1 = src1.reorder_if_differ_in(pd.src1_desc());
  // May replace dst's buffer and layout when they differ from pd.dst_desc().
  dst.reinit_if_possible(pd.dst_desc());

  dnnl::binary(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC_0, expected_src0},
       {DNNL_ARG_SRC_1, expected_src1},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
