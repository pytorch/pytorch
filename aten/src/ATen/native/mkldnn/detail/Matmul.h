#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::matmul_forward::compute (without bias).
// Deprecated 2-in-1 compute. Without bias.
void matmul_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f,
    const ideep::attr_t& attr = ideep::attr_t());

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
