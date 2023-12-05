#pragma once

#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/LRUCache.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

struct ReorderAttr {
 public:
  ReorderAttr(bool is_group = false)
      : pattr_(primitive_attr()),
        src_has_sc_(false),
        src_has_zp_(false),
        dst_has_sc_(false),
        dst_has_zp_(false) {}

 public:
  // [Note: Scale setting for reorder]
  // For no post op on reorder, dst = src_scale * src / dst_scale;
  // dst_scale should be set carefully.
  void set_src_sc_mask(int mask) {
    pattr_.set_scales_mask(DNNL_ARG_SRC, mask);
    src_has_sc_ = true;
  }

  void set_src_zp_mask(int mask) {
    pattr_.set_zero_points_mask(DNNL_ARG_SRC, mask);
    src_has_zp_ = true;
  }

  void set_dst_sc_mask(int mask) {
    pattr_.set_scales_mask(DNNL_ARG_DST, mask);
    dst_has_sc_ = true;
  }

  void set_dst_zp_mask(int mask) {
    pattr_.set_zero_points_mask(DNNL_ARG_DST, mask);
    dst_has_zp_ = true;
  }

  primitive_attr pattr() const {
    return pattr_;
  }

  bool src_has_sc() const {
    return src_has_sc_;
  }

  bool src_has_zp() const {
    return src_has_zp_;
  }

  bool dst_has_sc() const {
    return dst_has_sc_;
  }

  bool dst_has_zp() const {
    return dst_has_zp_;
  }

 private:
  primitive_attr pattr_;
  bool src_has_sc_;
  bool src_has_zp_;
  bool dst_has_sc_;
  bool dst_has_zp_;
};

static inline memory::desc check_group_and_create_plain_md(
    const Tensor& src,
    const Tensor& dst) {
  if (src.ndimension() == dst.ndimension()) {
    return memory::desc(
        get_onednn_dims(src),
        get_onednn_dtype_include_double(src),
        get_onednn_strides(src));
  } else if (
      ((src.ndimension() == dst.ndimension() - 1) &&
       (src.size(0) == dst.size(0) * dst.size(1))) ||
      ((src.ndimension() == dst.ndimension() + 1) &&
       (dst.size(0) == src.size(0) * src.size(1)))) {
    // group tensor
    return memory::desc(
        get_onednn_dims(dst),
        get_onednn_dtype_include_double(src),
        get_onednn_strides(dst.contiguous()));
  } else {
    TORCH_CHECK(0, "invalid src/dst dimension in oneDNN reorder ...");
  }
}

static inline void reorder(
    const Tensor& src,
    Tensor& dst,
    const ReorderAttr& rattr = ReorderAttr()) {
  RECORD_FUNCTION("dnnl_reorder", std::vector<c10::IValue>({src}));

  if (dst.is_same(src))
    return;

  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain()
      ? check_group_and_create_plain_md(src, dst)
      : src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_md = dst_ctx.is_plain()
      ? memory::desc(
            get_onednn_dims(dst),
            get_onednn_dtype_include_double(dst),
            get_onednn_strides(dst))
      : dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  primitive prim;
  prim = dnnl::reorder(src_mem, dst_mem);

  DPCPP_ONEDNN_EXEC(
      prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
}

static inline void quantized_reorder(
    const Tensor& src,
    Tensor& dst,
    float* src_scale,
    int32_t* src_zero_point,
    float* dst_scale,
    int32_t* dst_zero_point,
    memory::dims scale_zp_sz,
    memory::dims scale_zp_st,
    const ReorderAttr& rattr) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain()
      ? check_group_and_create_plain_md(src, dst)
      : src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_md = dst_ctx.is_plain()
      ? memory::desc(
            get_onednn_dims(dst),
            get_onednn_dtype_include_double(dst),
            get_onednn_strides(dst))
      : dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  std::unordered_map<int, memory> reorder_args;

  reorder_args.insert({DNNL_ARG_SRC, src_mem});
  reorder_args.insert({DNNL_ARG_DST, dst_mem});

  memory::desc src_sc_md, src_zp_md, dst_sc_md, dst_zp_md;
  memory src_sc_mem, src_zp_mem, dst_sc_mem, dst_zp_mem;

  if (rattr.src_has_sc()) {
    src_sc_md = memory::desc(scale_zp_sz, memory::data_type::f32, scale_zp_st);
    src_sc_mem = dpcpp_onednn_memory(src_sc_md, engine, src_scale);
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_mem});
  }

  if (rattr.src_has_zp()) {
    src_zp_md = memory::desc(scale_zp_sz, memory::data_type::s32, scale_zp_st);
    src_zp_mem = dpcpp_onednn_memory(src_zp_md, engine, src_zero_point);
    reorder_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem});
  }

  if (rattr.dst_has_sc()) {
    dst_sc_md = memory::desc(scale_zp_sz, memory::data_type::f32, scale_zp_st);
    dst_sc_mem = dpcpp_onednn_memory(src_sc_md, engine, dst_scale);
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_mem});
  }

  if (rattr.dst_has_zp()) {
    dst_zp_md = memory::desc(scale_zp_sz, memory::data_type::s32, scale_zp_st);
    dst_zp_mem = dpcpp_onednn_memory(dst_zp_md, engine, dst_zero_point);
    reorder_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem});
  }

  primitive prim;
  auto pattr = rattr.pattr();
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  // Here change scale to scale_md
  create_key(key, src_md, dst_md, src_sc_md, src_zp_md, dst_sc_md, dst_zp_md);
  prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem, pattr);
#else
  prim = dnnl::reorder(src_mem, dst_mem, pattr);
#endif

  DPCPP_ONEDNN_EXEC(prim, strm, reorder_args);
}

static inline void reorder_copy(const Tensor& src, Tensor& dst) {
  RECORD_FUNCTION("reorder_copy", std::vector<c10::IValue>({src}));
  xpu::oneDNN::reorder(src, dst);
}

} // namespace oneDNN
} // namespace xpu
