#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::binary::compute.
void binary(
    const ideep::tensor& src0,
    const ideep::tensor& src1,
    ideep::tensor& dst,
    dnnl::algorithm algorithm);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
