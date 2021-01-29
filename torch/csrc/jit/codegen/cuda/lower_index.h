#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API IndexLowering : public OptInDispatch {
 public:
  static std::vector<Expr*> getIndexedExprs(
      Fusion* fusion,
      std::vector<Expr*> incoming_exprs) {
    FUSER_PERF_SCOPE("IndexLowering::getIndexedExprs");
    FusionGuard fg(fusion);
    IndexLowering il;
    il.generate(incoming_exprs);
    return il.lowered_exprs;
  }

 private:
  IndexLowering();

  // Wrap pushBack, if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Open the for loop.
  void handle(kir::ForLoop*) final;

  // Open the for loop.
  void handle(kir::IfThenElse*) final;

  // Remake operations with TensorIndex
  void handle(UnaryOp*) final;
  void handle(BinaryOp*) final;
  void handle(TernaryOp*) final;
  void handle(ReductionOp*) final;
  void handle(BroadcastOp*) final;
  void handle(kir::Allocate*) final;
  void handle(kir::Sync*) final;

  void generate(const std::vector<Expr*>& exprs);

  Val* lowerOperand(Val* op, Val* out) const;
  Val* lowerOutput(Expr* expr) const;

 private:
  std::vector<Expr*> lowered_exprs;

  // This is a slight work around as scope has a couple definitions, we have the
  // Scope that's in ForLoop/IfThenElse which is really just a wrapper around
  // std::vector<Expr*> and then we have the actual ForLoop/IfThenElse. We want
  // to be able to carry both around because when we push back to a scope it
  // could be either the body or else body of the IfThenElse. However, we want
  // to understand the nesting of IfThenElse/ForLoop nodes.
  kir::Scope* active_scope = nullptr;
  Expr* active_scope_expr = nullptr;

  kir::IrBuilder ir_builder_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
