#pragma once

#include <c10/macros/Export.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class IrContainer;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer
class TORCH_CUDA_CU_API IrBuilderPasskey {
  friend class IrBuilder;

 public:
  // TODO: Collapse ir_container and Kernel once Kernel inherits from
  // IrContainer
  IrContainer* const ir_container_ = nullptr;

 private:
  explicit IrBuilderPasskey(IrContainer* ir_container)
      : ir_container_(ir_container) {}
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
