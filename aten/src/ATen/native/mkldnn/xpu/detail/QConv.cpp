
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

static std::tuple<
    dnnl::memory::desc,
    dnnl::memory::desc,
    dnnl::memory::desc,
    dnnl::memory::desc>
qconv_get_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    std::optional<at::Tensor> bias,
    const at::Tensor& dst,
    int64_t groups) {
  // create dnnl::memory desc from the src/wgh/dst tensors
  dnnl::memory::desc src_usr_md, wgh_usr_md, dst_usr_md, bias_usr_md;
  auto ndim = src.ndimension();
  bool src_is_cl =
      (src.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (src.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);
  auto fmt_src = conv_src_fmt(ndim, src_is_cl);

  auto src_tz = src.sizes().vec();
  auto src_data_t = get_onednn_dtype(src);
  src_usr_md = dnnl::memory::desc(src_tz, src_data_t, fmt_src);

  auto dst_tz = dst.sizes().vec();
  auto dst_data_t = get_onednn_dtype(dst);
  dst_usr_md = dnnl::memory::desc(dst_tz, dst_data_t, fmt_src);

  auto ic = src.size(1);
  auto oc = dst.size(1);
  auto wei_data_t = dnnl::memory::data_type::s8;
  bool wgh_is_cl =
      (wgh.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (wgh.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);
  dnnl::memory::dims wgh_tz =
      compatible_weight_dims(ndim, groups, oc, ic, wgh.sizes());
  auto fmt_wgh = conv_weight_fmt(ndim, groups != 1, wgh_is_cl);
  wgh_usr_md = dnnl::memory::desc(wgh_tz, wei_data_t, fmt_wgh);

  if (bias.has_value()) {
    bias_usr_md = dnnl::memory::desc(
        bias.value().sizes().vec(),
        dnnl::memory::data_type::f32,
        dnnl::memory::format_tag::x);
  }

  return {src_usr_md, wgh_usr_md, bias_usr_md, dst_usr_md};
}

at::Tensor quantized_convolution(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    at::Tensor output,
    double inv_output_scale,
    int64_t output_zero_point,
    std::optional<at::Tensor> accum,
    double accum_scale,
    int64_t accum_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<std::string_view> binary_attr,
    std::optional<at::Scalar> binary_alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm) {
  Attr attr = Attr(
      /*q_scale=*/static_cast<float>(1.0 / inv_output_scale),
      /*zp=*/output_zero_point);

  auto ndim = act.ndimension();
  construct_attr_by_post_op(
      binary_attr.has_value() ? binary_attr.value() : "none",
      binary_alpha.has_value() ? binary_alpha.value().to<double>() : 1.0,
      accum_scale,
      accum_zero_point,
      accum,
      unary_attr.has_value() ? unary_attr.value() : "none",
      unary_scalars,
      unary_algorithm.has_value() ? unary_algorithm.value() : "",
      attr);

  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "Quantized convolution only supports 3D, 4D, 5D tensor");
  TORCH_CHECK(
      output.defined(),
      "A valid output is required for quantized convolution.");

  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  // input tensors config
  dnnl::memory::dims src_dims = act.sizes().vec();
  dnnl::memory::dims weight_dims = weight.sizes().vec();
  // conv config
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding_front_top_left = padding.vec();
  dnnl::memory::dims _padding_back_bottom_right = padding.vec();
  dnnl::memory::dims _dilation = compatible_dilation(dilation);
  dnnl::post_ops po;
  // extract post ops
  po = attr.extract_post_ops(output);
  int mask_ac = 0, mask_weight;
  // [Note: Per-channel quantization mask setting]
  // Per-channel quantization is on weight output channel mostly, mask_weight=
  // 1 here means 2^0. 0 means the 0th dimension of weight tensor, aka output
  // channel. DNN requires mask = 2^k for the kth axis to be quantized. Only
  // one axis quantization is supported in XPU. Multi channel quantization
  // is not supported. In addition, src, output should still be per-tensor
  // quant, aka mask=0. Per-channel quantization on activation is not
  // supported in conv.
  mask_weight = weight_zero_points.numel() > 1 ? 1 : 0;
  if (groups > 1 && weight_zero_points.numel() > 1)
    mask_weight = (2 ^ 0) | (2 ^ 1); // 2^0 (group) | 2^1 (output channel)
  dnnl::primitive_attr pattr;

  bool src_need_zp = (act_zero_point != 0);
  bool dst_need_zp = (output_zero_point != 0);

  // create usr_md for tensors, and md for conv primitive
  auto [src_md, weight_md, bias_md, output_md] =
      qconv_get_md(act, weight, bias, output, groups);

  // get tensor md
  auto ic = act.size(1);
  auto oc = output.size(1);
  dnnl::memory::dims weight_tz =
      compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());

  pattr.set_scales_mask(DNNL_ARG_SRC, mask_ac);
  pattr.set_scales_mask(DNNL_ARG_DST, mask_ac);
  pattr.set_scales_mask(DNNL_ARG_WEIGHTS, mask_weight);
  pattr.set_post_ops(po);

  if (src_need_zp)
    pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
  if (dst_need_zp)
    pattr.set_zero_points_mask(DNNL_ARG_DST, mask_ac);
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // create primitive
  auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      bias.has_value() ? bias_md : dnnl::memory::desc(),
      output_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  dnnl::convolution_forward conv_forward =
      dnnl::convolution_forward(conv_fwd_pd);

  dnnl::memory src_m, weight_m, output_m, bias_m;

  src_m = make_onednn_memory(src_md, engine, act.data_ptr());
  output_m = make_onednn_memory(output_md, engine, output.data_ptr());
  weight_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  if (bias.has_value()) {
    bias_m = make_onednn_memory(bias_md, engine, bias.value().data_ptr());
  }

  std::unordered_map<int, dnnl::memory> args;
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, args);
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, weight_m});
  args.insert({DNNL_ARG_DST, output_m});
  if (bias.has_value()) {
    args.insert({DNNL_ARG_BIAS, bias_m});
  }

  dnnl::memory src_sc_m, src_zp_m;
  Tensor src_sc_tensor, src_zp_tensor;
  src_sc_m = dnnl_memory_from_host_scalar(
      static_cast<float>(act_scale), src_sc_tensor, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_m});
  if (src_need_zp) {
    src_zp_m = dnnl_memory_from_host_scalar(
        static_cast<int32_t>(act_zero_point), src_zp_tensor, engine);
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_m});
  }

  dnnl::memory dst_sc_m, dst_zp_m;
  Tensor dst_sc_tensor, dst_zp_tensor;
  dst_sc_m = dnnl_memory_from_host_scalar(
      static_cast<float>(inv_output_scale), dst_sc_tensor, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_m});
  if (dst_need_zp) {
    dst_zp_m = dnnl_memory_from_host_scalar(
        static_cast<int32_t>(output_zero_point), dst_zp_tensor, engine);
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_m});
  }

  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      act.options().dtype(at::kByte),
      std::nullopt);
  auto scratchpad_m = make_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // Weight scale is now tensor in nature, directly create dnnl::memory from it
  weight_scales = weight_scales.to(at::kFloat);
  dnnl::memory::desc weight_sc_md = dnnl::memory::desc(
      get_onednn_dims(weight_scales),
      dnnl::memory::data_type::f32,
      dnnl::memory::format_tag::x);
  dnnl::memory weight_sc_m =
      make_onednn_memory(weight_sc_md, engine, weight_scales.data_ptr());
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weight_sc_m});

  auto qconv_event = dnnl::sycl_interop::execute(conv_forward, stream, args);

  return output;
}

} // namespace at::native::onednn
