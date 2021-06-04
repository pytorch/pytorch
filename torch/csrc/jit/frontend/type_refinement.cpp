#include <torch/csrc/jit/frontend/type_refinement.h>

namespace torch {
namespace jit {

using Refinements = std::vector<Refinement>;

const std::string& Refinement::identifier() const {
  return identifier_;
}

TypePtr Refinement::type() const {
  return type_;
}

RefinementSet RefinementSet::And(const RefinementSet& rhs) const {
  // if the result of an AND is true, both a & b had to be true,
  // so we take the union of a.true_refinements and b.true_refinements.
  // if the result is false, either a or b could have been false,
  // so we take their intersection.
  return RefinementSet(
      unionSet(true_refinements_, rhs.true_refinements_),
      intersectSet(false_refinements_, rhs.false_refinements_));
}
RefinementSet RefinementSet::Or(const RefinementSet& rhs) const {
  // if the result of an OR is true, either a & b could have been true,
  // so we take the intersection of a.true_refinements & b.true_refinements.
  // if the result is false, both a and b had to be false,
  // so we take their union.
  return RefinementSet(
      intersectSet(true_refinements_, rhs.true_refinements_),
      unionSet(false_refinements_, rhs.false_refinements_));
}

RefinementSet RefinementSet::Not() const {
  return RefinementSet(false_refinements_, true_refinements_);
}
const std::vector<Refinement> RefinementSet::activeRefinements() const {
  return true_refinements_;
}

bool RefinementSet::sameVar(const Refinement& a, const Refinement& b) {
  return a.identifier() == b.identifier();
}
Refinements RefinementSet::unionSet(
    const Refinements& a,
    const Refinements& b) {
  Refinements result = a;
  for (const Refinement& r : b) {
    auto it =
        std::find_if(result.begin(), result.end(), [&](const Refinement& e) {
          return e.identifier() == r.identifier();
        });
    if (it == result.end()) {
      result.push_back(r);
    } else if (*it->type() != *r.type()) {
      // we only keep refinements when they exactly match one
      // refinement type, for instance, we do not attempt to refine:
      // isinstance(x, float) and isinstance(x, int)
      result.erase(it);
    }
  }
  return result;
}
Refinements RefinementSet::intersectSet(
    const Refinements& a,
    const Refinements& b) {
  Refinements result;
  for (const Refinement& r : a) {
    auto it = std::find_if(b.begin(), b.end(), [&](const Refinement& e) {
      return e.identifier() == r.identifier();
    });
    if (it != b.end() && r.type() == it->type()) {
      result.push_back(r);
    }
  }
  return result;
}

Value* CondValue::value() const {
  return value_;
}
const RefinementSet& CondValue::refinements() const {
  return refinements_;
}
c10::optional<bool> CondValue::staticIf() const {
  return static_if_;
}

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
