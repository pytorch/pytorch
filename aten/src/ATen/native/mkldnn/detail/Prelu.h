#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::prelu_forward::compute.
void prelu_forward(
    const ITensor& src,
    const ITensor& weight,
    ITensor& dst,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::prelu_backward::compute.
void prelu_backward(
    const ITensor& src,
    const ITensor& weight,
    const ITensor& diff_dst,
    ITensor& diff_src,
    ITensor& diff_weight);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
