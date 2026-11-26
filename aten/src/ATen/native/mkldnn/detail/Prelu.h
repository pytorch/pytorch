#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::prelu_forward::compute.
void prelu_forward(
    const MkldnnMemory& src,
    const MkldnnMemory& weight,
    MkldnnMemory& dst,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward);

// Ported from ideep::prelu_backward::compute.
void prelu_backward(
    const MkldnnMemory& src,
    const MkldnnMemory& weight,
    const MkldnnMemory& diff_dst,
    MkldnnMemory& diff_src,
    MkldnnMemory& diff_weight);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
