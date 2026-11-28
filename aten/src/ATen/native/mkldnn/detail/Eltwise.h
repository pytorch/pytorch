#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::eltwise_forward::compute.
void eltwise_forward(
    const MkldnnMemory& src,
    MkldnnMemory& dst,
    dnnl::algorithm algorithm = dnnl::algorithm::eltwise_relu,
    dnnl::prop_kind prop_kind = dnnl::prop_kind::forward,
    float alpha = 0.0,
    float beta = 0.0);

// Ported from ideep::eltwise_backward::compute.
void eltwise_backward(
    const MkldnnMemory& src,
    const MkldnnMemory& diff_dst,
    MkldnnMemory& diff_src,
    dnnl::algorithm algorithm = dnnl::algorithm::eltwise_relu,
    float alpha = 0.0,
    float beta = 0.0);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
