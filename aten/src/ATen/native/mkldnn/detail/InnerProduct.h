#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::inner_product_forward::compute (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    MkldnnMemory& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward(
    const MkldnnMemory& src,
    const MkldnnMemory& weights,
    MkldnnMemory& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (with bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const MkldnnMemory& src,
    const MkldnnMemory& other,
    const MkldnnMemory& weights,
    const MkldnnMemory& bias,
    MkldnnMemory& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::compute_binary (without bias).
template <bool reorder_src = true, bool reorder_weight = true>
void inner_product_forward_binary(
    const MkldnnMemory& src,
    const MkldnnMemory& other,
    const MkldnnMemory& weights,
    MkldnnMemory& dst,
    const Attr& attr = Attr(),
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::inner_product_forward::expected_weights_desc.
// The packed layout can depend on `attr` (e.g. fpmath or accumulation mode).
MkldnnMemory::desc inner_product_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    dnnl::memory::data_type x_dtype = dnnl::memory::data_type::f32,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward,
    const Attr& attr = Attr());

// Ported from ideep::inner_product_backward_data::compute.
void inner_product_backward_data(
    const MkldnnMemory& diff_dst,
    const MkldnnMemory& weights,
    const dnnl::memory::dims& diff_src_dims,
    MkldnnMemory& diff_src);

// Ported from ideep::inner_product_backward_weights::compute (with bias).
void inner_product_backward_weights(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    MkldnnMemory& diff_weights,
    MkldnnMemory& diff_bias);

// Ported from ideep::inner_product_backward_weights::compute (without bias).
void inner_product_backward_weights(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    MkldnnMemory& diff_weights);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
