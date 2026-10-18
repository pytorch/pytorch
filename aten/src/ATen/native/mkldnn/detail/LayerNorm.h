#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::layer_normalization_forward::compute.
void layer_normalization_forward(
    const ideep::tensor& src,
    const ideep::tensor& scale,
    const ideep::tensor& shift,
    ideep::tensor& dst,
    ideep::tensor& mean,
    ideep::tensor& variance,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
