#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::pooling_forward::compute.
void pooling_forward(
    const ideep::tensor& src,
    const dnnl::memory::dims& output_sizes,
    ideep::tensor& dst,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::algorithm algorithm,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::pooling_backward::compute.
void pooling_backward(
    const ideep::tensor& diff_dst,
    const ideep::tensor& dst,
    const ideep::tensor& src,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::algorithm algorithm);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
