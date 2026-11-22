#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::direct_copy::compute.
void direct_copy(const ideep::tensor& src, ideep::tensor& dst);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
