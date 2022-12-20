#pragma once
#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API PredicateElimination : public IterVisitor {
 public:
  void build(Fusion* fusion);

  //! True if expr does not need a predicate
  //!
  //! \param expr Tensor expression
  bool canOmitPredicate(const Expr* expr) const;

  //! Value to initialize out-of-bound regions
  Val* getInitValue(TensorView* tv) const;

  //! Dump to string for debugging
  std::string toString() const;

  // A utility to set removal info of `to` the same as `from`.
  //  See issue #1641
  // We build predicate info before lowering but more expressions
  //  are created during lowering that this class also need to
  //  keep track of to make sure correct predicate removal is
  //  applied.
  // This utility is a quick patch for the missing information
  //  since it might be better just to recompute predicate info
  //  if all expressions were mutated, but that'd take much more
  //  global info to reliably track.
  void propagateRemovalInfo(const Expr* from, const Expr* to);

 private:
  using IterVisitor::handle;

  void handle(Expr* expr) final;

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
