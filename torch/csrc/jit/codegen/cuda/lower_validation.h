#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Currently this does 3 things:
 *
 * (1) Run a validation pass on the IR making sure there are no mistakes or
 * unsupported scheduling.
 *
 * (2) Replace symbolic sizes for global memory with named scalars
 *     i.e. T0[i0] -> T0[T0.size[0]]
 *
 * (3) Change computeAt structure to make sure computeAt structure follows the
 * expression structure.
 *
 */

void TORCH_CUDA_API PrepareForLowering(Fusion* fusion);

// Compute at can have some circular references. Before we can call any tv
// with tv->getComputeAtAxis(i) we need to break those circular dependencies.
void IRFixComputeAt(Fusion* fusion);

// TensorViews are all based on symbolic sizes. When we first initialize them
// we don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void IRReplaceSizes();

} // namespace fuser
} // namespace jit
} // namespace torch
