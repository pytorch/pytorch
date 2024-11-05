#pragma once
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>

#include <ATen/core/grad_mode.h>
#include <c10/core/MemoryFormat.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_version.h>


#define ONEDNN_SUPPORT_DETERMINISTIC (DNNL_VERSION_MAJOR >=3 && DNNL_VERSION_MINOR >=4)

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

bool is_onednn_matmul_strides(
    const at::Tensor& tensor,
    bool is_dst = false);

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
    const at::Tensor& weight,
    bool is_transpose);

} // namespace at::native::onednn
