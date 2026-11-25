#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
// ideep headers use assert() without including <cassert>.
#include <cassert>
#include <ideep/tensor.hpp>

namespace at::native::mkldnn {

// oneDNN tensor (memory descriptor, buffer, scales/zero points, workspace).
// Alias for ideep::tensor until the tensor class is ported.
using MkldnnMemory = ideep::tensor;

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
