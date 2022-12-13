#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <fusion.h>
#include <ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TensorViews are all based on symbolic sizes. When we first initialize them
// we don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void replaceSymbolicSizes(Fusion*);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
