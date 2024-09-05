
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

static inline void construct_attr_by_post_op(
    const c10::string_view& binary_post_op,
    double binary_alpha,
    double input1_scale,
    int64_t input1_zero_point,
    // input1_desc,
    const c10::string_view& unary_post_op,
    const torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    const c10::string_view& unary_post_op_algorithm,
    Attr& attr
){
    if(binary_post_op == "none"){
        if( unary_post_op == "relu"){
            attr = attr.append_post_eltwise(
                /* eltwise_scale */ 1.f,
                /* alpha */ 0.f,
                /* beta */ 0.f,
                attr.kind_with_relu);
        }else if (unary_post_op == "leaky_relu"){
            auto alpha = unary_post_op_args[0].value().to<float>();
            attr = attr.append_post_eltwise(1.0, alpha, 0.f, attr.kind_with_relu);
        }else if (unary_post_op == "tanh"){
            attr = attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_tanh);
        }else if (unary_post_op == "gelu"){
            auto post_algorithm = unary_post_op_algorithm == "none" ?
                attr.kind_with_gelu_erf : attr.kind_with_gelu_tanh;
            attr = attr.append_post_eltwise(1.0f, 0.0f, 0.0f, post_algorithm);
        }else if (unary_post_op == "hardtanh"){
            auto alpha = unary_post_op_args[0].value().to<float>();
            auto beta = unary_post_op_args[1].value().to<float>();
            attr = attr.append_post_eltwise(1.0, alpha, beta, attr.kind_with_clip);
        }else if (unary_post_op == "hardswish"){
            attr = attr.append_post_eltwise(1.0f, 1.f / 6.f, 1.f / 2.f, attr.kind_with_hardswish);
        }else if (unary_post_op == "swish"){
            attr = attr.append_post_eltwise(1.0f, 1.0f, 0.0f, attr.kind_with_swish);
        }else {
            TORCH_CHECK(unary_post_op == "none", "onednn qlinear: unspported unary post op", unary_post_op);
        }
    }
}

static std::tuple<dnnl::memory::desc, dnnl::memory::desc, dnnl::memory::desc> qconv_get_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    bool is_channels_last_suggested) {
  // create dnnl::memory desc from the src/wgh/dst tensors
  dnnl::memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto fmt_src =
      conv_src_fmt(ndim, is_channels_last_suggested);

  auto src_tz = src.sizes().vec();
  auto src_data_t = get_onednn_dtype(src);
  src_usr_md = dnnl::memory::desc(src_tz, src_data_t, fmt_src);


  auto dst_tz = dst.sizes().vec();
  auto dst_data_t = get_onednn_dtype(dst);
  dst_usr_md = dnnl::memory::desc(dst_tz, dst_data_t, fmt_src);

  auto ic = src.size(1);
  auto oc = dst.size(1);
  auto wei_data_t = dnnl::memory::data_type::s8;
  dnnl::memory::dims wgh_tz =
      compatible_weight_dims(ndim, groups, oc, ic, wgh.sizes());
  auto fmt_wgh = conv_weight_fmt(
      ndim,
      groups != 1,
      is_channels_last_suggested);
  wgh_usr_md = dnnl::memory::desc(wgh_tz, wei_data_t, fmt_wgh);

  return {src_usr_md, wgh_usr_md, dst_usr_md};
}

at::Tensor quantized_convolution_pt2(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    c10::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    at::Tensor output,
    double inv_output_scale,
    int64_t output_zero_point,
    c10::optional<at::Tensor> accum,
    double accum_scale,
    int64_t accum_zero_point,
    c10::optional<c10::ScalarType> output_dtype,
    c10::optional<c10::string_view> binary_attr,
    c10::optional<at::Scalar> binary_alpha,
    c10::optional<c10::string_view> unary_attr,
    torch::List<c10::optional<at::Scalar>> unary_scalars,
    c10::optional<c10::string_view> unary_algorithm) {
  // TODO: use arg to create proper attr
  Attr attr = Attr(/*q_scale=*/ 1.0/inv_output_scale, /*zp=*/output_zero_point);

  construct_attr_by_post_op(
    binary_attr.has_value() ? binary_attr.value() : "none",
    binary_alpha.has_value() ? binary_alpha.value().to<double>() : 1.0,
    accum_scale,
    accum_zero_point,
    unary_attr.has_value() ? unary_attr.value() : "none",
    unary_scalars,
    unary_algorithm.has_value() ? unary_algorithm.value() : "",
    attr
  );

  auto ndim = act.ndimension();
  if(bias.has_value()){
    attr.append_bias(bias.value(), ndim-2);
  }
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution only supports 3D, 4D, 5D tensor");
  TORCH_CHECK(
      output.defined(), "Quantized convlution should always define output");

  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();

  // create usr_md for tensors, and md for conv primitive
  dnnl::memory::desc src_md, weight_md, output_md;
  bool is_channels_last_suggested = use_channels_last_for_conv(act, weight);
  // input tensors config
  dnnl::memory::dims src_dims = act.sizes().vec();
  dnnl::memory::dims weight_dims = weight.sizes().vec();
  auto src_data_t = get_onednn_dtype_include_double(act);
  auto dst_data_t = get_onednn_dtype_include_double(output);
  // conv config
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding_front_top_left = padding.vec();
  dnnl::memory::dims _padding_back_bottom_right = padding.vec();
  dnnl::memory::dims _dilation = compatible_dilation(dilation);
  dnnl::post_ops po;
  // extract post ops
  po = attr.extract_post_ops(output, /*is_quantized*/true);
  // set conv primitive scale and zero_point
  std::vector<float> conv_scale = {1};
  int mask_ac = 0, mask_weight;
  // [Note: Per-channel quantization mask setting]
  // Per-channel quantization is on weight output channel mostly, mask_weight=
  // 1 here means 2^0. 0 means the 0th dimension of weight tensor, aka output
  // channel. DNN requires mask = 2^k for the kth axis to be quantized. Only
  // one axis quantization is supported in IPEX. Multi channel quantization
  // is not supported. In addition, src, output should still be per-tensor
  // quant, aka mask=0. Per-channel quantization on activation is not
  // supported in conv.
  mask_weight = weight_zero_points.numel() > 1 ? 1 : 0;
  dnnl::primitive_attr pattr;

  bool src_need_zp = (act_scale != 0);
  bool weight_is_per_channel = (weight_scales.numel() > 0);

  dnnl::convolution_forward conv_forward;

  std::tie(src_md, weight_md, output_md) =
      qconv_get_md(act, weight, output, groups, is_channels_last_suggested);

  // get tensor md
  auto ic = act.size(1);
  auto oc = output.size(1);
  dnnl::memory::dims weight_tz =
      compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());

  pattr.set_scales_mask(DNNL_ARG_SRC, mask_ac);
  pattr.set_scales_mask(DNNL_ARG_WEIGHTS, mask_weight);
  pattr.set_post_ops(po);

  // Only setting zp mask when zp is not zero
  // See: [Note: Use symmetric quant implementation when zp is 0]
  if (src_need_zp)
    pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // create primitive
  auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      dnnl::memory::desc(),
      output_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  conv_forward = dnnl::convolution_forward(conv_fwd_pd);

  dnnl::memory src_m, weight_m, output_m;
  Tensor src_blocked, weight_blocked, output_blocked = output;

  src_m = make_onednn_memory(src_md, engine, act.data_ptr());
  output_m = make_onednn_memory(output_md, engine, output.data_ptr());
  weight_m = make_onednn_memory(weight_md, engine, weight.data_ptr());

  std::unordered_map<int, dnnl::memory> args;
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, args);
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, weight_m});
  args.insert({DNNL_ARG_DST, output_m});

  dnnl::memory src_sc_m, src_zp_m;
  Tensor src_sc_tensor, src_zp_tensor;
  src_sc_m = dnnl_memory_from_host_scalar(
      static_cast<float>(act_scale), src_sc_tensor, engine);
  src_zp_m = dnnl_memory_from_host_scalar(
      static_cast<int32_t>(act_zero_point), src_zp_tensor, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_m});

  // Only setting zp when zp is not zero
  // See: [Note: Use symmetric quant implementation when zp is 0]
  Tensor srz_zp;
  dnnl::memory::desc src_zp_md;
  if (src_need_zp) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_m});
  }

  // dst scale is no need for setting, since it is fused in postop via linear
  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, act.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = make_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // Weight scale is now tensor in nature, directly create dnnl::memory from it
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

}