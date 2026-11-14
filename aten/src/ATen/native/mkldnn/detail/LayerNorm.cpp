#include <ATen/native/mkldnn/detail/LayerNorm.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void layer_normalization_forward(
    const ideep::tensor& src,
    const ideep::tensor& scale,
    const ideep::tensor& shift,
    ideep::tensor& dst,
    ideep::tensor& mean,
    ideep::tensor& variance,
    float epsilon) {
  const auto& engine = ideep::engine::cpu_engine();
  const auto flags = dnnl::normalization_flags::use_scale |
      dnnl::normalization_flags::use_shift;
  auto src_desc = src.get_desc();
  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto pd = dnnl::layer_normalization_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      src_desc,
      src_desc,
      epsilon,
      flags,
      op_attr);

  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  mean.reinit_if_possible(pd.mean_desc());
  variance.reinit_if_possible(pd.variance_desc());
  dst.reinit_if_possible(pd.dst_desc());
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::layer_normalization_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC, expected_src},
       {DNNL_ARG_SCALE, scale},
       {DNNL_ARG_SHIFT, shift},
       {DNNL_ARG_MEAN, mean},
       {DNNL_ARG_VARIANCE, variance},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
