#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Replace with mutator as IndexLowering is replacing expr's with
// versions that are doing indexing
class TORCH_CUDA_CU_API IndexLowering : private OptOutConstDispatch {
 public:
  static std::vector<Expr*> getIndexedExprs(std::vector<Expr*> incoming_exprs) {
    FUSER_PERF_SCOPE("GpuLower::Lower::IndexLowering::getIndexedExprs");
    IndexLowering il;
    il.generate(incoming_exprs);
    return il.lowered_exprs_;
  }

 private:
  IndexLowering() = default;

  void pushBack(Expr*);

  // Return the most recently inserted
  //  expression in the current active
  //  scope or global scope.
  Expr* back() const;

  // Insert an expression before the current top-level expression.
  void insertAtTopLevel(Expr* expr);

  void handle(const ViewAsScalar*) final;
  void handle(const UnaryOp*) final;
  void handle(const BinaryOp*) final;
  void handle(const TernaryOp*) final;
  void handle(const ReductionOp*) final;
  void handle(const GroupedReductionOp*) final;
  void handle(const WelfordOp*) final;
  void handle(const MmaOp*) final;
  void handle(const BroadcastOp*) final;

  void handle(const kir::ForLoop*) final;
  void handle(const kir::IfThenElse*) final;
  void handle(const kir::Allocate*) final;
  void handle(const kir::BlockSync*) final;
  void handle(const kir::GridSync*) final;

  void generate(const std::vector<Expr*>& exprs);

  Val* lowerSrcIndex(Val* val, Val* dst) const;

  Val* lowerDstIndex(Val* dst) const;

  void handleBlockReduction(const ReductionOp* rop, Val* out, Val* in);
  void handleGridReduction(const ReductionOp* rop, Val* out, Val* in);

  void handleBlockReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);
  void handleGridReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);

  void handleGridWelford(WelfordOp* new_wop);

 private:
  std::vector<Expr*> lowered_exprs_;

  // This is a slight work around as scope has a couple definitions, we have the
  // Scope that's in ForLoop/IfThenElse which is really just a wrapper around
  // std::vector<Expr*> and then we have the actual ForLoop/IfThenElse. We want
  // to be able to carry both around because when we push back to a scope it
  // could be either the body or else body of the IfThenElse. However, we want
  // to understand the nesting of IfThenElse/ForLoop nodes.
  kir::Scope* active_scope_ = nullptr;

  // Track for loops to send to indexing. Similar to what's done in
  // kir::IrVisitor
  std::vector<kir::ForLoop*> for_loops_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
