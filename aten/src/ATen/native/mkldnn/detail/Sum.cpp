#include <ATen/native/mkldnn/detail/Sum.h>

#if AT_MKLDNN_ENABLED()

#include <unordered_map>

namespace at::native::mkldnn {

void sum(
    const std::vector<float>& scales,
    const std::vector<MkldnnMemory>& srcs,
    MkldnnMemory& dst) {
  const auto& engine = Engine::cpu_engine();
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
  MkldnnMemory scratchpad(pd.scratchpad_desc());
  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratchpad}};
  for (size_t i = 0; i < srcs.size(); ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + static_cast<int>(i), srcs[i]});
  }

  dnnl::sum(pd).execute(IStream::default_stream(), args);
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
