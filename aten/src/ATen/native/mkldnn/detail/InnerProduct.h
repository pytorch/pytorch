#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::inner_product_forward::compute (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const ideep::attr_t& attr = ideep::attr_t(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::expected_weights_desc.
ideep::tensor::desc inner_product_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    dnnl::memory::data_type x_dtype = dnnl::memory::data_type::f32,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_backward_data::compute.
void inner_product_backward_data(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src);

// Ported from ideep::inner_product_backward_weights::compute (with bias).
void inner_product_backward_weights(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias);

// Ported from ideep::inner_product_backward_weights::compute (without bias).
void inner_product_backward_weights(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_weights);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
