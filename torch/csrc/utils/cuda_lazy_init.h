#pragma once

#include <c10/core/TensorOptions.h>

// cuda_lazy_init() is always compiled, even for CPU-only builds.
// Thus, it does not live in the cuda/ folder.

namespace torch {
namespace utils {

// The INVARIANT is that this function MUST be called before you attempt
// to get a CUDA Type object from ATen, in any way.  Here are some common
// ways that a Type object may be retrieved:
//
//    - You call getNonVariableType or getNonVariableTypeOpt
//    - You call toBackend() on a Type
//
// It's important to do this correctly, because if you forget to add it
// you'll get an oblique error message about "Cannot initialize CUDA without
// ATen_cuda library" if you try to use CUDA functionality from a CPU-only
// build, which is not good UX.
//
void cuda_lazy_init();
void set_requires_cuda_init(bool value);

static void maybe_initialize_cuda(const at::TensorOptions& options) {
  if (options.device().is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

} // namespace utils
} // namespace torch
