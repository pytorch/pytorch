#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::convolution_backward_data::compute_v2.
void convolution_backward_data_compute_v2(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    const int groups,
    bool is_channels_last = false,
    ideep::attr_t op_attr = ideep::attr_t());

// Ported from ideep::convolution_backward_weights::compute_v2 (with bias).
void convolution_backward_weights_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::convolution_backward_weights::compute_v2 (without bias).
void convolution_backward_weights_compute_v2(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    const dnnl::memory::dims& diff_weights_dims,
    ideep::tensor& diff_weights,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& dilates,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    int groups,
    bool is_channels_last = false,
    const ideep::attr_t& attr = ideep::attr_t());

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
