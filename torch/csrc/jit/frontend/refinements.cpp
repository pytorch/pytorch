#include <torch/csrc/jit/frontend/refinements.h>

namespace torch {
namespace jit {

RefinementSet::RefinementSet(
    Refinements true_refinements,
    Refinements false_refinements)
    : true_refinements_(std::move(true_refinements)),
      false_refinements_(std::move(false_refinements)) {}

RefinementSet::RefinementSet(Refinement single)
    : RefinementSet({std::move(single)}, {}) {}

RefinementSet::RefinementSet(Refinement single_true, Refinement single_false)
    : RefinementSet(
          Refinements({std::move(single_true)}),
          Refinements({std::move(single_false)})) {}

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
} // namespace jit
} // namespace torch
