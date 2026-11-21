#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::softmax_forward::compute.
void softmax_forward(
    const ITensor& src,
    ITensor& dst,
    int softmax_axis,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
