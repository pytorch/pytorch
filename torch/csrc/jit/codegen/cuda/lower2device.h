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

  std::vector<kir::Allocate*> dynamic_allocations() {
    return dynamic_smem_allocations_;
  }

  std::vector<kir::Allocate*> static_allocations() {
    return static_smem_allocations_;
  }

  // Converts a Fusion IR value into the Kernel IR equivalent
  //
  // TODO(kir): revisit this interface
  //
  static Val* lowerValue(const Val* val);

  Val* getLowerValue(const Val* val);

 private:
  void lower();

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void buildSizesMap();

 private:
  // List of global buffers
  // Allocate nodes track if it needs to be initialized to 0
  std::vector<kir::Allocate*> global_allocations_;

  // List of dynamic shared memory buffers
  std::vector<kir::Allocate*> dynamic_smem_allocations_;

  // List of static shared memory buffers
  std::vector<kir::Allocate*> static_smem_allocations_;

  // Lowered IR
  std::vector<Expr*> lowered_exprs_;

  // Fusion IR node to Kernel IR node mapping
  std::unordered_map<const Val*, Val*> kir_map_;

  Fusion* fusion_ = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch
