#include <ATen/native/mkldnn/detail/Lstm.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void lstm_forward_inference(
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    ideep::tensor& dst_layer,
    ideep::tensor& dst_iter,
    ideep::tensor& dst_iter_c,
    bool reverse,
    dnnl::prop_kind prop_kind) {
  const auto& engine = ideep::engine::cpu_engine();
  auto direction = reverse ? dnnl::rnn_direction::unidirectional_right2left
                           : dnnl::rnn_direction::unidirectional_left2right;

  auto src_layer_desc = src_layer.get_desc();
  auto src_iter_desc = src_iter.get_desc();
  auto src_iter_c_desc = src_iter_c.get_desc();

  auto weights_layer_desc = weights_layer.get_desc().to_format_any();
  auto weights_iter_desc = weights_iter.get_desc().to_format_any();

  ideep::attr_t op_attr;
  if (src_layer.get_data_type() == dnnl::memory::data_type::u8) {
    weights_layer_desc = ideep::tensor::desc(
        weights_layer.get_dims(),
        dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any);
    weights_iter_desc = ideep::tensor::desc(
        weights_iter.get_dims(),
        dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any);
    // ATen never passes quantization parameters; these are ideep's defaults.
    op_attr.set_rnn_data_qparams(-1., -1);
    op_attr.set_rnn_weights_qparams(-1, std::vector<float>());
  }

  auto bias_desc = bias.get_desc();
  auto dst_layer_desc = dst_layer.get_desc();
  auto dst_iter_desc = dst_iter.get_desc();
  auto dst_iter_c_desc = dst_iter_c.get_desc();

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::lstm_forward::primitive_desc(
      engine,
      prop_kind,
      direction,
      src_layer_desc,
      src_iter_desc,
      src_iter_c_desc,
      weights_layer_desc,
      weights_iter_desc,
      bias_desc,
      dst_layer_desc,
      dst_iter_desc,
      dst_iter_c_desc,
      op_attr);

  auto expected_weights_layer =
      weights_layer.reorder_if_differ_in(pd.weights_layer_desc(), op_attr);
  auto expected_weights_iter =
      weights_iter.reorder_if_differ_in(pd.weights_iter_desc(), op_attr);
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::lstm_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC_LAYER, src_layer},
       {DNNL_ARG_SRC_ITER, src_iter},
       {DNNL_ARG_SRC_ITER_C, src_iter_c},
       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
       {DNNL_ARG_BIAS, bias},
       {DNNL_ARG_DST_LAYER, dst_layer},
       {DNNL_ARG_DST_ITER, dst_iter},
       {DNNL_ARG_DST_ITER_C, dst_iter_c},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

std::tuple<ideep::tensor::desc, ideep::tensor::desc>
lstm_forward_inference_expected_weights_desc(
    const dnnl::memory::dims& output_sizes,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    bool reverse) {
  const auto& engine = ideep::engine::cpu_engine();
  auto direction = reverse ? dnnl::rnn_direction::unidirectional_right2left
                           : dnnl::rnn_direction::unidirectional_left2right;

  auto src_layer_desc = src_layer.get_desc();
  auto src_iter_desc = src_iter.get_desc();
  auto src_iter_c_desc = src_iter_c.get_desc();

  auto weights_layer_desc = weights_layer.get_desc().to_format_any();
  auto weights_iter_desc = weights_iter.get_desc().to_format_any();

  ideep::attr_t op_attr;
  if (src_layer.get_data_type() == dnnl::memory::data_type::u8) {
    weights_layer_desc = ideep::tensor::desc(
        weights_layer.get_dims(),
        dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any);
    weights_iter_desc = ideep::tensor::desc(
        weights_iter.get_dims(),
        dnnl::memory::data_type::s8,
        dnnl::memory::format_tag::any);
    // ATen never passes quantization parameters; these are ideep's defaults.
    op_attr.set_rnn_data_qparams(-1., -1);
    op_attr.set_rnn_weights_qparams(-1, std::vector<float>());
  }

  auto bias_desc = bias.get_desc();
  ideep::tensor::desc dst_layer_desc(
      output_sizes, src_layer.get_data_type(), dnnl::memory::format_tag::tnc);

  auto pd = dnnl::lstm_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_inference,
      direction,
      src_layer_desc,
      src_iter_desc,
      src_iter_c_desc,
      weights_layer_desc,
      weights_iter_desc,
      bias_desc,
      dst_layer_desc,
      src_iter_desc,
      src_iter_c_desc,
      op_attr);

  return std::make_tuple(
      ideep::tensor::desc(pd.weights_layer_desc()),
      ideep::tensor::desc(pd.weights_iter_desc()));
}

dnnl::lstm_forward::primitive_desc lstm_forward_training_prepare(
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& dst_layer,
    const ideep::tensor& dst_iter,
    const ideep::tensor& dst_iter_c,
    bool reverse) {
  const auto& engine = ideep::engine::cpu_engine();
  auto direction = reverse ? dnnl::rnn_direction::unidirectional_right2left
                           : dnnl::rnn_direction::unidirectional_left2right;

  auto weights_layer_desc = weights_layer.get_desc().to_format_any();
  auto weights_iter_desc = weights_iter.get_desc().to_format_any();

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  return dnnl::lstm_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      direction,
      src_layer.get_desc(),
      src_iter.get_desc(),
      src_iter_c.get_desc(),
      weights_layer_desc,
      weights_iter_desc,
      bias.get_desc(),
      dst_layer.get_desc(),
      dst_iter.get_desc(),
      dst_iter_c.get_desc(),
      op_attr);
}

void lstm_forward_training(
    const dnnl::lstm_forward::primitive_desc& pd,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& workspace,
    ideep::tensor& dst_layer,
    ideep::tensor& dst_iter,
    ideep::tensor& dst_iter_c) {
  auto expected_weights_layer =
      weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
  auto expected_weights_iter =
      weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

  dst_layer.reinit_if_possible(pd.dst_layer_desc());
  dst_iter.reinit_if_possible(pd.dst_iter_desc());
  dst_iter_c.reinit_if_possible(pd.dst_iter_c_desc());
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::lstm_forward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC_LAYER, src_layer},
       {DNNL_ARG_SRC_ITER, src_iter},
       {DNNL_ARG_SRC_ITER_C, src_iter_c},
       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
       {DNNL_ARG_BIAS, bias},
       {DNNL_ARG_DST_LAYER, dst_layer},
       {DNNL_ARG_DST_ITER, dst_iter},
       {DNNL_ARG_DST_ITER_C, dst_iter_c},
       {DNNL_ARG_WORKSPACE, workspace},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

void lstm_backward(
    const dnnl::lstm_forward::primitive_desc& forward_hints,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& weights_layer,
    const ideep::tensor& weights_iter,
    const ideep::tensor& bias,
    const ideep::tensor& dst_layer,
    const ideep::tensor& dst_iter,
    const ideep::tensor& dst_iter_c,
    const ideep::tensor& diff_dst_layer,
    const ideep::tensor& diff_dst_iter,
    const ideep::tensor& diff_dst_iter_c,
    const ideep::tensor& workspace,
    ideep::tensor& diff_src_layer,
    ideep::tensor& diff_src_iter,
    ideep::tensor& diff_src_iter_c,
    ideep::tensor& diff_weights_layer,
    ideep::tensor& diff_weights_iter,
    ideep::tensor& diff_bias,
    bool reverse) {
  if (is_avx2_vnni_2() &&
      one_of(
          src_layer.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = ideep::engine::cpu_engine();
  constexpr auto f32 = dnnl::memory::data_type::f32;
  auto direction = reverse ? dnnl::rnn_direction::unidirectional_right2left
                           : dnnl::rnn_direction::unidirectional_left2right;

  auto src_layer_desc = src_layer.get_desc();
  auto src_iter_desc = src_iter.get_desc();
  auto src_iter_c_desc = src_iter_c.get_desc();

  auto bias_desc = bias.get_desc();
  auto dst_layer_desc = dst_layer.get_desc();
  auto dst_iter_desc = dst_iter.get_desc();
  auto dst_iter_c_desc = dst_iter_c.get_desc();

  auto weights_layer_desc = weights_layer.get_desc().to_format_any();
  auto weights_iter_desc = weights_iter.get_desc().to_format_any();

  // Gradients are always computed in f32.
  auto diff_src_layer_desc = src_layer_desc.to_type(f32);
  auto diff_src_iter_desc = src_iter_desc.to_type(f32);
  auto diff_src_iter_c_desc = src_iter_c_desc.to_type(f32);
  auto diff_weights_layer_desc = ideep::tensor::desc(
      weights_layer.get_dims(), f32, dnnl::memory::format_tag::any);
  auto diff_weights_iter_desc = ideep::tensor::desc(
      weights_iter.get_dims(), f32, dnnl::memory::format_tag::any);
  auto diff_bias_desc = bias_desc.to_type(f32);
  auto diff_dst_layer_desc = dst_layer_desc.to_type(f32);
  auto diff_dst_iter_desc = dst_iter_desc.to_type(f32);
  auto diff_dst_iter_c_desc = dst_iter_c_desc.to_type(f32);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::lstm_backward::primitive_desc(
      engine,
      dnnl::prop_kind::backward,
      direction,
      src_layer_desc,
      src_iter_desc,
      src_iter_c_desc,
      weights_layer_desc,
      weights_iter_desc,
      bias_desc,
      dst_layer_desc,
      dst_iter_desc,
      dst_iter_c_desc,
      diff_src_layer_desc,
      diff_src_iter_desc,
      diff_src_iter_c_desc,
      diff_weights_layer_desc,
      diff_weights_iter_desc,
      diff_bias_desc,
      diff_dst_layer_desc,
      diff_dst_iter_desc,
      diff_dst_iter_c_desc,
      forward_hints,
      op_attr);

  auto expected_weights_layer =
      weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
  auto expected_weights_iter =
      weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

  diff_src_layer.reinit_if_possible(pd.diff_src_layer_desc());
  diff_src_iter.reinit_if_possible(pd.diff_src_iter_desc());
  diff_src_iter_c.reinit_if_possible(pd.diff_src_iter_c_desc());

  // oneDNN accumulates into the weight and bias gradients, so start from zero.
  ideep::tensor expected_diff_weights_layer;
  expected_diff_weights_layer.zero_init(pd.diff_weights_layer_desc());
  ideep::tensor expected_diff_weights_iter;
  expected_diff_weights_iter.zero_init(pd.diff_weights_iter_desc());
  ideep::tensor expected_diff_bias;
  expected_diff_bias.zero_init(pd.diff_bias_desc());
  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::lstm_backward(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_SRC_LAYER, src_layer},
       {DNNL_ARG_SRC_ITER, src_iter},
       {DNNL_ARG_SRC_ITER_C, src_iter_c},
       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
       {DNNL_ARG_BIAS, bias},
       {DNNL_ARG_DST_LAYER, dst_layer},
       {DNNL_ARG_DST_ITER, dst_iter},
       {DNNL_ARG_DST_ITER_C, dst_iter_c},
       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
       {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c},
       {DNNL_ARG_DIFF_WEIGHTS_LAYER, expected_diff_weights_layer},
       {DNNL_ARG_DIFF_WEIGHTS_ITER, expected_diff_weights_iter},
       {DNNL_ARG_DIFF_BIAS, expected_diff_bias},
       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
       {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c},
       {DNNL_ARG_WORKSPACE, workspace},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  diff_weights_layer.feed_from(expected_diff_weights_layer);
  diff_weights_iter.feed_from(expected_diff_weights_iter);
  diff_bias.feed_from(expected_diff_bias);
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
