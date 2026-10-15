#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::prelu_forward::compute.
void prelu_forward(
    const ideep::tensor& src,
    const ideep::tensor& weight,
    ideep::tensor& dst,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::prelu_backward::compute.
void prelu_backward(
    const ideep::tensor& src,
    const ideep::tensor& weight,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_src,
    ideep::tensor& diff_weight);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
