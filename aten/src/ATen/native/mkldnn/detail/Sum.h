#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::sum::compute.
void sum(
    const std::vector<float>& scales,
    const std::vector<MkldnnMemory>& srcs,
    MkldnnMemory& dst);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
