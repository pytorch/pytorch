#pragma once
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Update predicates with valid bool conditionals
//!
std::vector<kir::Expr*> generateConditionalFromPredicate(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs);

class TORCH_CUDA_CU_API PredicateElimination : public IterVisitor {
 public:
  void build(Fusion* fusion);

  //! True if expr does not need a predicate
  //!
  //! \param expr Tensor expression
  bool canOmitPredicate(const Expr* expr) const;

  //! True if expr does not need a predicate
  //!
  //! \param expr KIR tensor expr
  bool canOmitPredicate(const kir::Expr* expr) const;

  //! Value to initialize out-of-bound regions
  kir::Val* getInitValue(TensorView* tv) const;

  //! Dump to string for debugging
  std::string toString() const;

 private:
  using IterVisitor::handle;

  void handle(Expr* expr) override;

  //! Set a value to initialize out-of-bound regions
  bool setDefaultInitValue(TensorView* tv);
  //! Set a value to initialize out-of-bound regions of reduction tensors
  bool setReductionInitValue(TensorView* tv, Val* reduction_init);

  //! Check if expr needs to be predicated
  bool needsPredicate(Expr* expr) const;

 private:
  //! Expressions that are found to be safe without predicates
  std::unordered_set<const Expr*> non_predicated_exprs_;
  //! Tensors and their initialization values
  std::unordered_map<TensorView*, Val*> init_value_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
