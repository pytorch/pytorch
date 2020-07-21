#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Currently this does the following:
 *
 * (1) Run a validation pass on the IR making sure there are no mistakes or
 * unsupported scheduling.
 *
 * (2) Creates a mapping for symbolic sizes to named scalars
 *     i.e. T0[i0] -> T0[T0.size[0]]
 *
 * (3) Change computeAt structure to make sure computeAt structure follows the
 * expression structure.
 *
 * (4) Adjust TensorView memory types to make sure they are valid
 */

void TORCH_CUDA_API PrepareForLowering(Fusion* fusion);

// TensorViews are all based on symbolic sizes. When we first initialize them we
// don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void IrBuildSizesMap(Fusion* fusion);

// Adjust memory types to make sure they are valid
void IrAdjustMemoryTypes(Fusion* fusion);

} // namespace fuser
} // namespace jit
} // namespace torch
