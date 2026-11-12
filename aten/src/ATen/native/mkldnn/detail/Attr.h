#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep/tensor.hpp>

namespace at::native::mkldnn {

// Primitive attributes (post-ops, scales, zero points, modes).
using Attr = ideep::attr_t;

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
