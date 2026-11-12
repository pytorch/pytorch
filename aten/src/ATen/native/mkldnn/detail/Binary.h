#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::sum::compute.
void sum(
    const std::vector<float>& scales,
    const std::vector<ITensor>& srcs,
    ITensor& dst);

// Ported from ideep::binary::compute.
void binary(
    const ITensor& src0,
    const ITensor& src1,
    ITensor& dst,
    dnnl::algorithm algorithm);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
