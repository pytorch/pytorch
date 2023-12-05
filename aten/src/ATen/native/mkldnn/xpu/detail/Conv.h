#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>

#include <oneDNN/Runtime.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;
using namespace at::AtenIpexTypeQuantizedXPU;

namespace xpu {
namespace oneDNN {

constexpr int src_batch_size_dim = 0;
constexpr int wgh_dst_channels_dim = 0;

static inline memory::dims conv_dst_tz(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  memory::dims dst_tz(ndim);
  dst_tz[0] = src_tz[src_batch_size_dim];
  dst_tz[1] = wgh_tz[wgh_dst_channels_dim];
  for (size_t d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (wgh_tz[d] - 1) + 1;
    dst_tz[d] =
        (src_tz[d] +
         (padding_front_top_left[d - 2] + padding_back_bottom_right[d - 2]) -
         kernel) /
            stride[d - 2] +
        1;
  }
  return dst_tz;
}

static inline memory::dims compatible_dilation(IntArrayRef& dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? memory::format_tag::ncw
        : ((ndim == 4) ? memory::format_tag::nchw
                       : ((ndim == 5) ? memory::format_tag::ncdhw
                                      : memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? memory::format_tag::nwc
        : ((ndim == 4) ? memory::format_tag::nhwc
                       : ((ndim == 5) ? memory::format_tag::ndhwc
                                      : memory::format_tag::undef));
  }
}

static inline memory::format_tag conv_wgh_fmt(
    const int64_t ndim,
    const bool grouped = false,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::goiw : memory::format_tag::oiw)
        : (ndim == 4)
        ? (grouped ? memory::format_tag::goihw : memory::format_tag::oihw)
        : ((ndim == 5) ? (grouped ? memory::format_tag::goidhw
                                  : memory::format_tag::oidhw)
                       : memory::format_tag::undef);
  } else {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::gowi : memory::format_tag::owi)
        : (ndim == 4)
        ? (grouped ? memory::format_tag::gohwi : memory::format_tag::ohwi)
        : ((ndim == 5) ? (grouped ? memory::format_tag::godhwi
                                  : memory::format_tag::odhwi)
                       : memory::format_tag::undef);
  }
}

static inline memory::dims compatible_wgh_dims(
    const int64_t ndim,
    const int64_t groups,
    const int64_t oc,
    const int64_t ic,
    const IntArrayRef wsizes) {
  if (ndim == 3) {
    auto kw = wsizes[2];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kw})
                         : memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

static std::tuple<
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc>
conv_get_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    int memory_layout) {
  // create memory desc from the src/wgh/dst tensors
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto fmt_src =
      conv_src_fmt(ndim, memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
  if (src_ctx.is_plain()) {
    auto src_tz = src.sizes().vec();
    auto src_data_t = get_onednn_dtype_include_double(src);
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
  } else {
    src_usr_md = src_ctx.meta();
  }

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  if (dst_ctx.is_plain()) {
    auto dst_tz = dst.sizes().vec();
    auto dst_data_t = get_onednn_dtype_include_double(dst);
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    dst_usr_md = dst_ctx.meta();
  }

  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  if (wgh_ctx.is_plain()) {
    auto ic = src.size(1);
    auto oc = dst.size(1);
    auto wei_data_t = get_onednn_dtype_include_double(wgh);
    memory::dims wgh_tz =
        compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
    auto fmt_wgh = conv_wgh_fmt(
        ndim,
        groups != 1,
        memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  } else {
    wgh_usr_md = wgh_ctx.meta();
  }

  // create memory desc for conv primitive and query the blocked format
  memory::desc src_md, wgh_md, dst_md;
  if (memory_layout == MEMORY_LAYOUT_FOR_CONV::Blocked) {
    auto fmt_any = memory::format_tag::any;
    src_md = src.size(1) == 3
        ? src_usr_md
        : memory::desc(
              src_usr_md.get_dims(), src_usr_md.get_data_type(), fmt_any);
    wgh_md = memory::desc(
        wgh_usr_md.get_dims(), wgh_usr_md.get_data_type(), fmt_any);
    dst_md = memory::desc(
        dst_usr_md.get_dims(), dst_usr_md.get_data_type(), fmt_any);
  } else {
    src_md = src_usr_md;
    wgh_md = wgh_usr_md;
    dst_md = dst_usr_md;
  }
  return {src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md};
}

static memory conv_get_expected_src_memory(
    const at::Tensor& src,
    at::Tensor& src_blocked,
    memory::desc& src_usr_md,
    memory::desc& expected_src_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory src_m;
  if (src_usr_md != expected_src_md) {
    src_blocked =
        empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
    src_m =
        dpcpp_onednn_memory(expected_src_md, engine, src_blocked.data_ptr());
    if (need_reorder)
      xpu::oneDNN::reorder(src, src_blocked);
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    src_blocked = src;
  }
  return src_m;
}

static memory conv_get_expected_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_blocked,
    memory::desc& wgh_usr_md,
    memory::desc& expected_wgh_md,
    dnnl::engine& engine,
    bool weight_cache_optimization,
    bool need_reorder = true) {
  memory wgh_m;
  if (wgh_usr_md != expected_wgh_md) {
    wgh_blocked =
        empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    wgh_m =
        dpcpp_onednn_memory(expected_wgh_md, engine, wgh_blocked.data_ptr());

    if (need_reorder) {
      auto reshaped_wgh = wgh;
      // reshape for group convolution weight
      if (wgh_blocked.ndimension() > wgh.ndimension()) {
        // for groups conv case:
        // expected_wgh will be 5-D Tensor based on expected_wgh_md:
        // g/o/i/h/w or g/o/h/w/i
        // wgh will be 4-D Tensor based on PyTorch
        // (g)o/i/h/w or (g)o/h/w/i
        // we need to manually reshape 4-D wgh to 5-D,
        // consistent with expected_wgh
        reshaped_wgh = share_storage_and_set_strided_as(
            wgh,
            wgh_blocked.sizes(),
            /*compatible with different strides of weight (including contiguous,
               channels_last and non-contiguous) */
            compatible_groups_conv_strides(wgh, wgh_blocked.sizes().vec()),
            c10::nullopt);
      }
      xpu::oneDNN::reorder(reshaped_wgh, wgh_blocked);

      if (weight_cache_optimization) {
        auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_blocked);
        wgh_opt_ctx.set_aten_meta(
            {reshaped_wgh.sizes().vec(), reshaped_wgh.strides().vec()});
        DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
      }
    }
  } else {
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    wgh_blocked = wgh;
  }
  return wgh_m;
}

static memory conv_get_expected_dst_memory(
    const at::Tensor& dst,
    at::Tensor& dst_blocked,
    memory::desc& dst_usr_md,
    memory::desc& expected_dst_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory dst_m;
  if (dst_usr_md != expected_dst_md) {
    dst_blocked =
        empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, dst_blocked.data_ptr());

    if (need_reorder)
      xpu::oneDNN::reorder(dst, dst_blocked);
  } else {
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    dst_blocked = dst;
  }
  return dst_m;
}

static at::Tensor convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(src, wgh, /*is_transposed*/ false);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create usr_md for tensors, and md for conv primitive
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
      conv_get_md(src, wgh, dst, groups, memory_layout_for_conv);

  auto bia_fmt = memory::format_tag::x;
  auto bia_md = bia.defined()
      ? memory::desc(
            {dst.size(1)}, get_onednn_dtype_include_double(bia), bia_fmt)
      : memory::desc();

  // create conv primitive descriptor
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  // extract post ops
  primitive_attr pattr;
  post_ops po;
  attr.extract_post_ops(po, dst);
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (src_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

  auto conv_fwd_pd = convolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  auto weight_cache_optimization = [&]() {
    return memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked &&
        !at::GradMode::is_enabled();
  }();

  memory src_m, wgh_m, dst_m, bia_m;
  Tensor src_blocked, wgh_blocked, dst_blocked = dst;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_fwd_pd.src_desc();
    auto expected_wgh_md = conv_fwd_pd.weights_desc();
    auto expected_dst_md = conv_fwd_pd.dst_desc();
    src_m = conv_get_expected_src_memory(
        src, src_blocked, src_usr_md, expected_src_md, engine);
    wgh_m = conv_get_expected_wgh_memory(
        wgh,
        wgh_blocked,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        weight_cache_optimization);
    dst_m = conv_get_expected_dst_memory(
        dst, dst_blocked, dst_usr_md, expected_dst_md, engine, attr.with_sum());
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  }

  std::unordered_map<int, memory> args;
  if (bia.defined()) {
    bia_m = dpcpp_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }
  auto expected_dst_md = conv_fwd_pd.dst_desc();
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, po, args);

  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wgh_m});
  args.insert({DNNL_ARG_DST, dst_m});

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  auto conv_forward = convolution_forward(conv_fwd_pd);
  DPCPP_ONEDNN_EXEC(conv_forward, strm, args);

  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

static void convolution_backward_weights(
    at::Tensor& diff_wgh,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_wgh_aten_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(src, diff_dst, /*is_transposed=*/false);

  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
      conv_get_md(src, diff_wgh, diff_dst, groups, memory_layout_for_conv);
  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = diff_bia.defined()
      ? memory::desc({diff_dst.size(1)}, src_md.get_data_type(), bia_fmt)
      : memory::desc();

  // create fwd primitive hint
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  primitive_attr pattr;
  if (src_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  auto conv_fwd_pd = convolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  // create bwd weight primitive
  auto conv_bwd_w_pd = convolution_backward_weights::primitive_desc(
      engine,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      conv_fwd_pd,
      pattr);

  // create bwd memory
  Tensor expected_src, expected_diff_dst, expected_diff_wgh;
  memory src_m, diff_dst_m, diff_wgh_m;
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_bwd_w_pd.src_desc();
    auto expected_dst_md = conv_bwd_w_pd.diff_dst_desc();
    auto expected_wgh_md = conv_bwd_w_pd.diff_weights_desc();
    src_m = conv_get_expected_src_memory(
        src, expected_src, src_usr_md, expected_src_md, engine);
    diff_wgh_m = conv_get_expected_wgh_memory(
        diff_wgh,
        expected_diff_wgh,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false, // weight_cache
        false); // need_reorder
    diff_dst_m = conv_get_expected_dst_memory(
        diff_dst, expected_diff_dst, dst_usr_md, expected_dst_md, engine);

  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
    diff_wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, diff_wgh.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m});
  if (diff_bia.defined()) {
    memory diff_bia_m =
        dpcpp_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }
#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_bwd_w_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  // execute primitive
  auto conv_bwd_w = dnnl::convolution_backward_weights(conv_bwd_w_pd);
  DPCPP_ONEDNN_EXEC(conv_bwd_w, strm, args);

  if (is_onednn_layout_suggested && diff_wgh_m.get_desc() != wgh_usr_md) {
    // expected_diff_wgh contains the result of gw backward in blk format.
    // In training mode, plain gw output is expected for sgd update
    // Thus, we need one additional reorder here to make diff_wgh plain.
    auto reshaped_diff_wgh = diff_wgh;
    if (expected_diff_wgh.ndimension() > diff_wgh.ndimension()) {
      // for groups conv case:
      // expected_diff_wgh will be 5-D Tensor based on expected_diff_wgh_md:
      // g/o/i/h/w or g/o/h/w/i
      // diff_wgh will be 4-D Tensor based on PyTorch
      // (g)o/i/h/w or (g)o/h/w/i
      // we need to manually reshape 5-D expected_diff_wgh to 4-D,
      // consistent with PyTorch diff_wgh
      reshaped_diff_wgh = share_storage_and_set_strided_as(
          diff_wgh,
          expected_diff_wgh.sizes(),
          compatible_groups_conv_strides(
              diff_wgh, expected_diff_wgh.sizes().vec()),
          c10::nullopt);
    }
    xpu::oneDNN::reorder(expected_diff_wgh, reshaped_diff_wgh);
  }
}

static void convolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(diff_dst, weight, /*is_transposed=*/false);

  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
      conv_get_md(diff_src, weight, diff_dst, groups, memory_layout_for_conv);
  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = bias_defined
      ? memory::desc({diff_dst.size(1)}, wgh_md.get_data_type(), bia_fmt)
      : memory::desc();

  // create fwd primitive desc hint
  primitive_attr pattr;
  if (dst_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  auto conv_forward_pd = convolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  auto conv_backward_data_pd = convolution_backward_data::primitive_desc(
      engine,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      conv_forward_pd,
      pattr);

  // create memory
  Tensor expected_src, expected_wei, expected_dst;
  memory diff_dst_m, wei_m, diff_src_m;
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_backward_data_pd.diff_src_desc();
    auto expected_wgh_md = conv_backward_data_pd.weights_desc();
    auto expected_dst_md = conv_backward_data_pd.diff_dst_desc();
    diff_src_m = conv_get_expected_src_memory(
        diff_src, expected_src, src_usr_md, expected_src_md, engine, false);
    wei_m = conv_get_expected_wgh_memory(
        weight,
        expected_wei,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false); // weight_cache
    diff_dst_m = conv_get_expected_dst_memory(
        diff_dst, expected_dst, dst_usr_md, expected_dst_md, engine);
  } else {
    diff_src_m = dpcpp_onednn_memory(src_usr_md, engine, diff_src.data_ptr());
    wei_m = dpcpp_onednn_memory(wgh_usr_md, engine, weight.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_backward_data_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      conv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // execute primitive
  auto conv_backward_data =
      dnnl::convolution_backward_data(conv_backward_data_pd);
  DPCPP_ONEDNN_EXEC(conv_backward_data, strm, args);

  // propagate blk format
  if (is_onednn_layout_suggested &&
      diff_src.data_ptr() != expected_src.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(expected_src);
    DPCPPTensorContext::set_tensor_ctx(diff_src, std::move(blk_ctx));
  }
}

} // namespace oneDNN
} // namespace xpu
