#pragma once

#include <torch/csrc/Export.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API IndexLowering : private kir::IrVisitor {
 public:
  static std::vector<kir::Expr*> getIndexedExprs(
      std::vector<kir::Expr*> incoming_exprs) {
    FUSER_PERF_SCOPE("GpuLower::Lower::IndexLowering::getIndexedExprs");
    IndexLowering il;
    il.generate(incoming_exprs);
    return il.lowered_exprs_;
  }

 private:
  IndexLowering();

  void pushBack(kir::Expr*);

  void visit(const kir::ForLoop*) final;
  void visit(const kir::IfThenElse*) final;
  void visit(const kir::UnaryOp*) final;
  void visit(const kir::BinaryOp*) final;
  void visit(const kir::TernaryOp*) final;
  void visit(const kir::ReductionOp*) final;
  void visit(const kir::WelfordOp*) final;
  void visit(const kir::BroadcastOp*) final;
  void visit(const kir::Allocate*) final;
  void visit(const kir::Sync*) final;

  void generate(const std::vector<kir::Expr*>& exprs);

  kir::Val* lowerSrcIndex(kir::Val* val, kir::Val* dst) const;
  kir::Val* lowerDstIndex(kir::Val* dst) const;

 private:
  std::vector<kir::Expr*> lowered_exprs_;

  // This is a slight work around as scope has a couple definitions, we have the
  // Scope that's in ForLoop/IfThenElse which is really just a wrapper around
  // std::vector<Expr*> and then we have the actual ForLoop/IfThenElse. We want
  // to be able to carry both around because when we push back to a scope it
  // could be either the body or else body of the IfThenElse. However, we want
  // to understand the nesting of IfThenElse/ForLoop nodes.
  kir::Scope* active_scope_ = nullptr;
  kir::Expr* active_scope_expr_ = nullptr;

  kir::IrBuilder ir_builder_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
