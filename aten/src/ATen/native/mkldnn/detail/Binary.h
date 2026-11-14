#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::sum::compute.
void sum(
    const std::vector<float>& scales,
    const std::vector<ideep::tensor>& srcs,
    ideep::tensor& dst);

// Ported from ideep::binary::compute.
void binary(
    const ideep::tensor& src0,
    const ideep::tensor& src1,
    ideep::tensor& dst,
    dnnl::algorithm algorithm);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
