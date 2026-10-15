#include <ATen/native/mkldnn/detail/Softmax.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void softmax_forward(
    const ideep::tensor& src,
    ideep::tensor& dst,
    int softmax_axis,
    dnnl::prop_kind prop_kind) {
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc = src.get_desc();
  // May replace dst's buffer and layout when they differ from src's.
  dst.reinit_if_possible(src_desc);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::softmax_forward::primitive_desc(
      engine,
      prop_kind,
      dnnl::algorithm::softmax_accurate,
      src_desc,
      src_desc,
      softmax_axis,
      op_attr);
  ideep::tensor scratchpad(pd.scratchpad_desc());
  dnnl::softmax_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC, src},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
