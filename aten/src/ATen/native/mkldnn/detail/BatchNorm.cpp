#include <ATen/native/mkldnn/detail/BatchNorm.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void batch_normalization_forward_inference(
    const ITensor& src,
    const ITensor& mean,
    const ITensor& variance,
    const ITensor& scale,
    const ITensor& shift,
    ITensor& dst,
    float epsilon) {
  const auto& engine = Engine::cpu_engine();
  const auto pd_flags = dnnl::normalization_flags::use_scale |
      dnnl::normalization_flags::use_shift |
      dnnl::normalization_flags::use_global_stats;

  // Workaround for oneDNN issue intel/mkl-dnn#588 (kept from ideep).
  auto src_desc = src._get_unblocked_desc_if_4c_blocked();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto pd = dnnl::batch_normalization_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_inference,
      src_desc,
      src_desc,
      epsilon,
      pd_flags,
      op_attr);

  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  // May replace dst's buffer and layout when they differ from pd.dst_desc().
  dst.reinit_if_possible(pd.dst_desc());
  ITensor scratchpad(pd.scratchpad_desc());

  auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
  auto expected_var = variance.reorder_if_differ_in(pd.variance_desc());
  dnnl::batch_normalization_forward(pd).execute(
      Stream::default_stream(),
      {{DNNL_ARG_SRC, expected_src},
       {DNNL_ARG_SCALE, scale},
       {DNNL_ARG_SHIFT, shift},
       {DNNL_ARG_VARIANCE, expected_var},
       {DNNL_ARG_MEAN, expected_mean},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

void batch_normalization_forward_training(
    const ITensor& src,
    const ITensor& scale,
    const ITensor& shift,
    ITensor& dst,
    ITensor& mean,
    ITensor& variance,
    float epsilon) {
  const auto& engine = Engine::cpu_engine();
  const auto pd_flags = dnnl::normalization_flags::use_scale |
      dnnl::normalization_flags::use_shift;

  // Workaround for oneDNN issue intel/mkl-dnn#588 (kept from ideep).
  auto src_desc = src._get_unblocked_desc_if_4c_blocked();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::batch_normalization_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      src_desc,
      src_desc,
      epsilon,
      pd_flags,
      op_attr);

  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  mean.reinit_if_possible(pd.mean_desc());
  variance.reinit_if_possible(pd.variance_desc());
  dst.reinit_if_possible(pd.dst_desc());
  ITensor scratchpad(pd.scratchpad_desc());

  dnnl::batch_normalization_forward(pd).execute(
      Stream::default_stream(),
      {{DNNL_ARG_SRC, expected_src},
       {DNNL_ARG_SCALE, scale},
       {DNNL_ARG_SHIFT, shift},
       {DNNL_ARG_MEAN, mean},
       {DNNL_ARG_VARIANCE, variance},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

void batch_normalization_backward(
    const ITensor& src,
    const ITensor& mean,
    const ITensor& variance,
    const ITensor& diff_dst,
    const ITensor& scale,
    ITensor& diff_src,
    ITensor& diff_scale,
    ITensor& diff_shift,
    float epsilon) {
  if (is_avx2_vnni_2() &&
      one_of(
          src.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = Engine::cpu_engine();
  const auto pd_flags = dnnl::normalization_flags::use_scale |
      dnnl::normalization_flags::use_shift;
  // Workaround for oneDNN issue intel/mkl-dnn#588 (kept from ideep).
  auto src_desc = src._get_unblocked_desc_if_4c_blocked();
  auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      src_desc,
      src_desc,
      epsilon,
      pd_flags);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::batch_normalization_backward::primitive_desc(
      engine,
      dnnl::prop_kind::backward,
      forward_hints.src_desc(),
      forward_hints.dst_desc(),
      src_desc,
      epsilon,
      pd_flags,
      forward_hints,
      op_attr);

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
  auto expected_variance = variance.reorder_if_differ_in(pd.variance_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());
  diff_scale.reinit_if_possible(scale.get_desc());
  diff_shift.reinit_if_possible(scale.get_desc());

  ITensor scratchpad(pd.scratchpad_desc());

  dnnl::batch_normalization_backward(pd).execute(
      Stream::default_stream(),
      {{DNNL_ARG_SRC, expected_src},
       {DNNL_ARG_DIFF_DST, expected_diff_dst},
       {DNNL_ARG_SCALE, scale},
       {DNNL_ARG_MEAN, expected_mean},
       {DNNL_ARG_VARIANCE, expected_variance},
       {DNNL_ARG_DIFF_SRC, diff_src},
       {DNNL_ARG_DIFF_SCALE, diff_scale},
       {DNNL_ARG_DIFF_SHIFT, diff_shift},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
