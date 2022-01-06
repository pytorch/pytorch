#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_container.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace kir {
class Kernel;
}

class IrCloner;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer
class TORCH_CUDA_CU_API IrBuilderPasskey {
  friend class IrBuilder;
  friend class kir::IrBuilder;

 public:
  // TODO: Collapse ir_container and Kernel once Kernel inherits from
  // IrContainer
  IrContainer* const ir_container_ = nullptr;
  kir::Kernel* const kernel = nullptr;

 private:
  explicit IrBuilderPasskey(kir::Kernel* kernel);
  explicit IrBuilderPasskey(IrContainer* ir_container)
      : ir_container_(ir_container) {}
};

//! IR builder interface
class TORCH_CUDA_CU_API IrBuilder {
 public:
  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(Args&&... args) {
    auto container = FusionGuard::getCurFusion();
    // return create<T>(container, std::forward<Args>(args)...);
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(IrContainer* container, Args&&... args) {
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Clone an IR node, forwarding the arguments to the IrCloner constructor.
  //! Register clones with IrCloner's target container.
  template <class T>
  static T* clone(const T* src, IrCloner* ir_cloner);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
