#include <torch/csrc/jit/frontend/type_refinement.h>

namespace torch {
namespace jit {

RefinementSet findIsNoneRefinements(
    const Expr& lhs,
    Value* lhs_value,
    const Expr& rhs,
    Value* rhs_value,
    int tok) {
  if (rhs.kind() != TK_NONE && lhs.kind() == TK_NONE) {
    // make 'None is var' into 'var is None'
    return findIsNoneRefinements(rhs, rhs_value, lhs, lhs_value, tok);
  }
  if (rhs.kind() != TK_NONE || lhs.kind() != TK_VAR) {
    return {};
  }
  // statement must be var {is, is not} None
  auto name = Var(lhs).name().name();
  // XXX - while it should in theory be possible to specialize
  // the `x is None` to know x has type NoneType, we have previously not
  // done this. Unfortunately, doing this will make the type None
  // propagate further in all loaded models. The handling of
  // unwrap_optional will fail in these cases since export did
  // not expect that the input would be none and an unannotated None.
  // To enable this, we need to (1) implement a real casting operator
  // annotated(T, X) that stays in the graph and does the cast
  // and (2) only enable this OPTIONAL_NONE when loading newer
  // graphs because it is incompatible with older graphs.
  // Refinement none(name, RefinementKind::OPTIONAL_NONE);
  if (auto optional_type = lhs_value->type()->cast<OptionalType>()) {
    Refinement present(name, optional_type->getElementType());
    if (tok == TK_IS) {
      return RefinementSet({}, {present});
    } else { // TK_ISNOT
      return RefinementSet({present}, {});
    }
  }
  return RefinementSet();
}

} // namespace jit
} // namespace torch
