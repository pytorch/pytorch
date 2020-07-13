#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <map>
#include <ostream>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

// TODO: Change lowering so it can be called multiple times. It would be good to
// keep user references intact so they can lower it as they describe the kernel.
// Right now we can only lower once.

struct TORCH_CUDA_API GPULower : public OptOutMutator {
 private:
  Fusion* const fusion_;
  std::vector<Expr*> lowered_exprs;
  Expr* active_scope = nullptr;

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Custom dispatch for Expr, want to find out of it's a TV op
  Statement* mutate(Expr*) final;

  // Open the for loop.
  Statement* mutate(ForLoop*) final;

  // Open the for loop.
  Statement* mutate(IfThenElse*) final;

  // Remake operations with TensorIndex
  Statement* mutate(UnaryOp*) final;
  Statement* mutate(BinaryOp*) final;
  Statement* mutate(TernaryOp*) final;
  Statement* mutate(ReductionOp*) final;
  Statement* mutate(BroadcastOp*) final;

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void replaceSizes();
  void fixComputeAt(Fusion* fusion);

 public:
  // Init printer on ostream
  GPULower(Fusion* _fusion) : fusion_(_fusion) {}

  // print generated code to ostream
  std::vector<Expr*> getLoweredExprs();
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");
};

} // namespace fuser
} // namespace jit
} // namespace torch
