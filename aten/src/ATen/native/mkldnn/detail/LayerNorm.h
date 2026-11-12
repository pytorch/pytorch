#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::layer_normalization_forward::compute.
void layer_normalization_forward(
    const ITensor& src,
    const ITensor& scale,
    const ITensor& shift,
    ITensor& dst,
    ITensor& mean,
    ITensor& variance,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
