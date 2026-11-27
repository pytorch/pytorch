#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::layer_normalization_forward::compute.
void layer_normalization_forward(
    const MkldnnMemory& src,
    const MkldnnMemory& scale,
    const MkldnnMemory& shift,
    MkldnnMemory& dst,
    MkldnnMemory& mean,
    MkldnnMemory& variance,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
