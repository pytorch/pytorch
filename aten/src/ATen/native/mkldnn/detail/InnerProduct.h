#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::inner_product_forward::compute (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const ITensor& src,
    const ITensor& weights,
    const ITensor& bias,
    ITensor& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const ITensor& src,
    const ITensor& weights,
    ITensor& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const ITensor& src,
    const ITensor& other,
    const ITensor& weights,
    const ITensor& bias,
    ITensor& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const ITensor& src,
    const ITensor& other,
    const ITensor& weights,
    ITensor& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::expected_weights_desc.
ITensor::desc inner_product_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    dnnl::memory::data_type x_dtype = dnnl::memory::data_type::f32,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_backward_data::compute.
void inner_product_backward_data(
    const ITensor& diff_dst,
    const ITensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ITensor& diff_src);

// Ported from ideep::inner_product_backward_weights::compute (with bias).
void inner_product_backward_weights(
    const ITensor& src,
    const ITensor& diff_dst,
    ITensor& diff_weights,
    ITensor& diff_bias);

// Ported from ideep::inner_product_backward_weights::compute (without bias).
void inner_product_backward_weights(
    const ITensor& src,
    const ITensor& diff_dst,
    ITensor& diff_weights);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
