#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
// ideep headers use assert() without including <cassert>.
#include <cassert>
#include <ideep/tensor.hpp>

namespace at::native::mkldnn {

// oneDNN tensor (memory descriptor, buffer, scales/zero points, workspace).
using ITensor = ideep::tensor;

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
