#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API GpuLower {
  class KernelIrMapper;

 public:
  GpuLower() = default;

  explicit GpuLower(Fusion* fusion) : fusion_(fusion) {
    lower();
  }

  // print generated code to ostream
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");

  std::string getKernel(const std::string& kernel_name = "CUDAGeneratedKernel");

  std::vector<kir::Allocate*> global_allocations() {
    return global_allocations_;
  }

  std::vector<kir::Allocate*> sync_allocations() {
    return sync_allocations_;
  }

  // Converts a Fusion IR value into the Kernel IR equivalent
  //
  // TODO(kir): revisit this interface
  //
  static Val* lowerValue(const Val* val);

 private:
  void lower();

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void buildSizesMap();

  // Adjust memory types to make sure they are valid
  void adjustMemoryTypes();

 private:
  // List of global buffers (not including buffers for grid syncronization)
  std::vector<kir::Allocate*> global_allocations_;

  // List of syncronization buffers that must be initialized to 0 when running
  // the fusion
  std::vector<kir::Allocate*> sync_allocations_;

  // Lowered IR
  std::vector<Expr*> lowered_exprs_;

  // Fusion IR node to Kernel IR node mapping
  std::unordered_map<const Val*, Val*> kir_map_;

  Fusion* fusion_ = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch
