#include <c10/xpu/XPUFunctions.h>
#include <ATen/ATen.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>

namespace at::native::onednn {

static inline dnnl::memory::dims deconv_compatible_dilation(IntArrayRef& dilation) {
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline std::vector<int64_t> compatible_groups_deconv_strides(
    const at::Tensor& weight,
    dnnl::memory::dims group_size) {
  std::vector<int64_t> strides = weight.strides().vec();
  strides[0] = weight.strides()[1];
  strides[1] = weight.strides()[0];
  strides.insert(strides.begin(), group_size[2] * weight.strides()[0]);
  return strides;
}

dnnl::memory::dims deconv_dst_size(
    IntArrayRef src_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef dst_padding,
    int64_t groups) {
  auto dim = src_size.size();
  dnnl::memory::dims dst_size(dim);
  auto kernel_size = weight_size.slice(2);

  dst_size[0] = src_size[0];
  dst_size[1] = weight_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    dst_size[d] = (src_size[d] - 1) * stride[d - 2] - 2 * padding[d - 2] +
        (dilation[d - 2] * (kernel_size[d - 2] - 1) + 1) + dst_padding[d - 2];
  }
  return dst_size;
}

static inline dnnl::memory::format_tag deconv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  // 3D: n/c/w (n/w/c)         [a/b/c (a/c/b)]
  // 4D: n/c/h/w (n/h/w/c)     [a/b/c/d (a/c/d/b)]
  // 5D: n/c/d/h/w (n/d/h/w/c) [a/b/c/d/e (a/c/d/e/b)]
  if (!is_channels_last) {
    return (ndim == 3)
        ? dnnl::memory::format_tag::ncw
        : ((ndim == 4) ? dnnl::memory::format_tag::nchw
                       : ((ndim == 5) ? dnnl::memory::format_tag::ncdhw
                                      : dnnl::memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? dnnl::memory::format_tag::nwc
        : ((ndim == 4) ? dnnl::memory::format_tag::nhwc
                       : ((ndim == 5) ? dnnl::memory::format_tag::ndhwc
                                      : dnnl::memory::format_tag::undef));
  }
}

static inline std::vector<int64_t> deconv_weight_fmt(
    const at::Tensor& weight,
    const int64_t ndim,
    dnnl::memory::dims weight_size,
    const bool grouped = false,
    const bool is_channels_last = false) {
  // 3D fmt: (g)i/o/w ((g)i/w/o)  [b/a/c  (b/c/a)]
  // 4D fmt: (g)i/o/h/w ((g)i/h/w/o) [b/a/c/d (b/c/d/a)]
  // 5D fmt: (g)i/o/d/h/w ((g)i/d/h/w/o) [b/a/c/d/e (b/c/d/e/a)]
  auto strides_ = weight.strides().vec();
  std::vector<int64_t> strides;
  if (grouped) {
    strides = compatible_groups_deconv_strides(weight, weight_size);
  } else {
    strides = strides_;
    std::swap(strides[0], strides[1]);
  }
  return strides;
}

static inline dnnl::memory::dims deconv_compatible_weight_dims(
    int64_t ndim,
    int64_t groups,
    int64_t oc,
    int64_t ic,
    IntArrayRef weight_size) {
  if (ndim == 3) {
    auto kw = weight_size[2];
    return (groups != 1) ? dnnl::memory::dims({groups, oc / groups, ic / groups, kw})
                         : dnnl::memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = weight_size[2];
    auto kw = weight_size[3];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : dnnl::memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = weight_size[2];
    auto kh = weight_size[3];
    auto kw = weight_size[4];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : dnnl::memory::dims({oc, ic, kd, kh, kw});
  } else {
    TORCH_CHECK(0, "unsupported dimension in xpu oneDNN deconvolution...");
  }
}

static std::tuple<
    dnnl::memory::desc,
    dnnl::memory::desc,
    dnnl::memory::desc>
deconv_get_plain_md(
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& dst,
    int64_t groups,
    bool is_channels_last_suggested) {
  auto ndim = src.ndimension();
  auto src_data_t = get_onednn_dtype_include_double(src);
  auto fmt_src = deconv_src_fmt(ndim, is_channels_last_suggested);
  auto src_usr_md = dnnl::memory::desc(src.sizes().vec(), src_data_t, fmt_src);

  auto dst_data_t = get_onednn_dtype_include_double(dst);
  auto dst_usr_md = dnnl::memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);

  auto ic = src.size(1);
  auto oc = dst.size(1);
  dnnl::memory::dims weight_size =
      deconv_compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  auto weight_dt = get_onednn_dtype_include_double(weight);
  auto fmt_weight = deconv_weight_fmt(
      weight, ndim, weight_size, groups != 1, is_channels_last_suggested);
  dnnl::memory::desc weight_usr_md = dnnl::memory::desc(weight_size, weight_dt, fmt_weight);

  return {src_usr_md, weight_usr_md, dst_usr_md};
}

sycl::event deconvolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();

  bool is_channels_last_suggested = use_channels_last_for_conv(src, weight, /*is_transposed=*/true);

  // create usr_md for tensors, and md for conv primitive
  auto [src_md, weight_md, dst_md] =
      deconv_get_plain_md(src, weight, dst, groups, is_channels_last_suggested);

  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bia_md = bia.defined()
      ? dnnl::memory::desc(
            {dst.size(1)}, get_onednn_dtype_include_double(bia), bia_fmt)
      : dnnl::memory::desc();

  // create primitive desc
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding = padding.vec();
  dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);

  // construct primitive attr
  dnnl::primitive_attr pattr;
  dnnl::post_ops po = attr.extract_post_ops(dst);
  pattr.set_post_ops(po);
  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        pattr.set_deterministic(true);
  #endif

  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  dnnl::memory src_m, weight_m, dst_m, bia_m;
  at::Tensor src_blocked, weight_blocked, dst_blocked = dst;

  src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  weight_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  dst_m = make_onednn_memory(dst_md, engine, dst.data_ptr());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, weight_m});
  args.insert({DNNL_ARG_DST, dst_m});

  if (bia.defined()) {
    auto bia_m = make_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }
  if (attr.with_binary())
    attr.construct_post_binary(deconv_fwd_pd, args);

  size_t scratchpad_size = deconv_fwd_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = make_onednn_memory(
      deconv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  auto deconv_fwd = dnnl::deconvolution_forward(deconv_fwd_pd);
  sycl::event deconv_event = dnnl::sycl_interop::execute(deconv_fwd, stream, args, deps);
  return deconv_event;

}

sycl::event deconvolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();

  bool is_channels_last_suggested =
      use_channels_last_for_conv(diff_dst, weight, /*is_transposed=*/true);
  // create memory desc
  auto [src_md, weight_md, dst_md] =
      deconv_get_plain_md(
          diff_src, weight, diff_dst, groups, is_channels_last_suggested);

  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bias_md = bias_defined
      ? dnnl::memory::desc({diff_dst.size(1)}, weight_md.get_data_type(), bia_fmt)
      : dnnl::memory::desc();

  // create fwd primitive desc hint
  dnnl::primitive_attr pattr;
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        pattr.set_deterministic(true);
  #endif

  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding = padding.vec();
  dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);
  auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bias_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  // create bwd primitive desc
  auto deconv_backward_data_pd = dnnl::deconvolution_backward_data::primitive_desc(
      engine,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      deconv_fwd_pd);

  // create memory
  dnnl::memory diff_dst_m, wei_m, diff_src_m;

  diff_src_m = make_onednn_memory(src_md, engine, diff_src.data_ptr());
  wei_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());

  // insert args
  std::unordered_map<int, dnnl::memory> args;
  size_t scratchpad_size = deconv_backward_data_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      deconv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // execute primitive
  auto deconv_backward_data =
      dnnl::deconvolution_backward_data(deconv_backward_data_pd);
  sycl::event deconv_bwd_data_event = dnnl::sycl_interop::execute(deconv_backward_data, stream, args, deps);
  return deconv_bwd_data_event;

}

sycl::event deconvolution_backward_weights(
    at::Tensor& diff_weight,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();

  bool is_channels_last_suggested =
      use_channels_last_for_conv(src, diff_dst, /*is_transposed=*/true);

  // create memory desc
  auto [src_md, weight_md, dst_md] = deconv_get_plain_md(
          src, diff_weight, diff_dst, groups, is_channels_last_suggested);

  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bia_md = diff_bia.defined()
      ? dnnl::memory::desc({diff_dst.size(1)}, src_md.get_data_type(), bia_fmt)
      : dnnl::memory::desc();

  // create fwd primitive desc hint
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding = padding.vec();
  dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);
  dnnl::primitive_attr pattr;

  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        pattr.set_deterministic(true);
  #endif
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  auto deconv_bwd_w_pd = dnnl::deconvolution_backward_weights::primitive_desc(
      engine,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      deconv_fwd_pd,
      pattr);

  // create bwd dnnl::memory
  dnnl::memory src_m, diff_dst_m, diff_weight_m;

  src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());
  diff_weight_m = make_onednn_memory(weight_md, engine, diff_weight.data_ptr());

  // insert args
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_weight_m});

  if (diff_bia.defined()) {
    dnnl::memory diff_bia_m =
        make_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }

  size_t scratchpad_size = deconv_bwd_w_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = make_onednn_memory(
      deconv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // execute primitive
  auto deconv_bwd_w = dnnl::deconvolution_backward_weights(deconv_bwd_w_pd);

  sycl::event deconv_bwd_w_event = dnnl::sycl_interop::execute(deconv_bwd_w, stream, args, deps);
  return deconv_bwd_w_event;

}

} // namespace at::native::onednn
