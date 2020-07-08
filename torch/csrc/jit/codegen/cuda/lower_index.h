#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API IndexLowering : public OptOutMutator {
 private:
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
  void generate(const std::vector<Expr*>& exprs);

 public:
  static std::vector<Expr*> getIndexedExprs(
      Fusion* fusion,
      std::vector<Expr*> incoming_exprs) {
    FusionGuard fg(fusion);
    IndexLowering il;
    il.generate(incoming_exprs);
    return il.lowered_exprs;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
