#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/arg_spec.h"
#include "torch/csrc/jit/fuser/fused_kernel.h"

#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fuser {

// Performs device-independent "upfront" compilation of the given fusion_group
// Returns a key that can be used to run the fusion later
TORCH_API int64_t registerFusion(const Node* fusion_group);

// Performs device-specific "runtime" compilation of the given kernel
//  with the runtime arguments specified in ArgSpec.
//  Outputs are allocated using map_size on the specified device.
TORCH_API std::shared_ptr<FusedKernel> compileKernel(
  const KernelSpec& spec
, const ArgSpec& arg_spec
, const std::vector<int64_t>& map_size
, const int device);

TORCH_API size_t nCompiledKernels();

} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER || USE_CPU_FUSER
