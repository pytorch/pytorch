#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/mutator.h>

namespace torch {
namespace jit {
namespace fuser {

const Statement* BaseMutator::mutate(
    const Statement* const statement) {
  if (statement->isVal())
    return mutate(static_cast<const Val*>(statement));
  else if (statement->isExpr())
    return mutate(static_cast<const Expr*>(statement));
  else
    throw std::runtime_error("Could not detect type in mutate(const Statement*).");
}
const Statement* BaseMutator::mutate(const Val* const val) {
  return val->dispatch_mutator(this);
}

const Statement* BaseMutator::mutate(const Expr* const expr) {
  return expr->dispatch_mutator(this);
}

const Statement* BaseMutator::mutate(const Float* const f) {
  return f;
}

const Statement* BaseMutator::mutate(const Int* const i) {
  return i;
}

const Statement* BaseMutator::mutate(const UnaryOp* const uop) {
  const Val* out = static_cast<const Val*>(uop->out()->dispatch_mutator(this));
  const Val* in = static_cast<const Val*>(uop->in()->dispatch_mutator(this));
  // TODO CHECK IF ADD CHANGED, RETURN NEW ONE.
  if 
  (
    !(
         out->same_as(uop->out())
      && in->same_as(uop->in())
    )
  )
    return new UnaryOp(uop->type(), out, in);
  return uop;
}

const Statement* BaseMutator::mutate(const BinaryOp* const bop) {
  const Val* out = static_cast<const Val*>(bop->out()->dispatch_mutator(this));
  const Val* lhs = static_cast<const Val*>(bop->lhs()->dispatch_mutator(this));
  const Val* rhs = static_cast<const Val*>(bop->rhs()->dispatch_mutator(this));
  if
  (
    !(
         out != bop->out()
      && lhs != bop->lhs()
      && rhs != bop->rhs()
    )
  )
    return new BinaryOp(bop->type(), out, lhs, rhs);
  return bop;
}

void BaseMutator::mutate(Fusion* fusion) {
  std::vector<const Expr*> new_exprs;
  std::vector<const Expr*> orig_exprs = fusion->exprs();

  for (std::vector<const Expr*>::size_type i = 0; i < orig_exprs.size(); i++) {
    const Statement* new_stmt = orig_exprs[i]->dispatch_mutator(this);
    assert(new_stmt->isExpr());
    new_exprs.push_back(static_cast<const Expr*>(new_stmt));
  }

  for (std::vector<const Expr*>::size_type i = 0; i < orig_exprs.size(); i++) {
    if (orig_exprs[i] != new_exprs[i]) {
      fusion->removeExpr(orig_exprs[i]);
    }
  }
}

const Statement* BaseMutator::mutate(const TensorDomain* const t) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const TensorView* const t) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const IterDomain* const t) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const Tensor* const t) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const Split* const split) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const Merge* const merge) {
  throw std::runtime_error("Not implemented yet.");
}

const Statement* BaseMutator::mutate(const Reorder* const reorder) {
  throw std::runtime_error("Not implemented yet.");
}

} // namespace fuser
} // namespace jit
} // namespace torch
