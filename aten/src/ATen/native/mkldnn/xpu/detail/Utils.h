#pragma once
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <iostream>

#include <ATen/core/grad_mode.h>
#include <c10/core/MemoryFormat.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_version.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

#define ONEDNN_SUPPORT_DETERMINISTIC \
  (DNNL_VERSION_MAJOR >= 3 && DNNL_VERSION_MINOR >= 4)

namespace at::native::onednn {

dnnl::memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last = false,
    bool allow_undef = false);

dnnl::memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef = false);

dnnl::memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,
    bool allow_undef = false);

bool is_supported_onednn_dtype(const at::Tensor& tensor);

dnnl::memory::dims get_onednn_dims(const at::Tensor& tensor);

dnnl::memory::dims get_onednn_strides(const at::Tensor& tensor);
dnnl::memory::desc get_onednn_md(const at::Tensor& tensor);

bool onednn_strides_check(const at::Tensor& src);
bool is_broadcast(const at::Tensor& t);

bool is_onednn_matmul_strides(const at::Tensor& tensor, bool is_dst = false);

bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other);

at::MemoryFormat get_cl_tag_by_ndim(const int64_t ndim);

bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other,
    bool is_fusion = false);

bool use_channels_last_for_conv(
    const at::Tensor& src,
    const at::Tensor& weight);

dnnl::memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false);

dnnl::memory::dims compatible_weight_dims(
    const int64_t ndim,
    const int64_t groups,
    const int64_t oc,
    const int64_t ic,
    const IntArrayRef wsizes);

dnnl::memory::format_tag conv_weight_fmt(
    const int64_t ndim,
    const bool grouped = false,
    const bool is_channels_last = false);

template <typename Vec>
dnnl::memory::dims compatible_dilation(Vec&& dilation) {
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

template <typename T>
dnnl::memory dnnl_memory_from_host_scalar(
    T host_value,
    Tensor& holder,
    dnnl::engine& engine) {
  auto options = at::TensorOptions()
                     .dtype(c10::CppTypeToScalarType<T>::value)
                     .device(kXPU);
  holder = at::empty({1}, options).fill_(host_value);
  dnnl::memory::desc md = get_onednn_md(holder);
  dnnl::memory mem = make_onednn_memory(md, engine, holder.data_ptr());
  return mem;
}

} // namespace at::native::onednn
