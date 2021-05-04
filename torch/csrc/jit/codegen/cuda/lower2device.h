#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <memory>
#include <ostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API GpuLower {
  class KernelIrMapper;

 public:
  GpuLower() = default;

  explicit GpuLower(Fusion* fusion) : fusion_(fusion) {
    lower();
  }

  Kernel* kernel() const;

  // Converts a Fusion IR value into the Kernel IR equivalent
  //
  // TODO(kir): revisit this interface
  //
  static Val* lowerValue(const Val* val);

  // TODO(kir): we have two methods which do almost the same thing
  //
  Val* getLowerValue(const Val* val);

  //! Returns the currently active lowering object
  //! (or nullptr if no lowering is in progress)
  static GpuLower* current();

 private:
  void lower();

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void replaceSymbolicSizes();

 private:
  // Lowered Kernel IR
  std::unique_ptr<Kernel> kernel_;

  // Fusion IR node to Kernel IR node mapping
  std::unordered_map<const Val*, Val*> kir_map_;

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
