#pragma once

#include <string>
#include <vector>
#include "ATen/core/jit_type.h"

/**
 * This file contains helpers for tracking type refinements.
 *
 * When we encounter a conditional expression that tells us information about a
 * type statically, we want to be able to "refine" the type to be more specific
 * in the bodies of the if-else blocks.
 *
 *   # a: Optional[Tensor]
 *   if a is None:
 *      # we know a: None
 *   else:
 *      # we know a: Tensor
 */
namespace torch {
namespace jit {

/**
 * Tracks the refined type for the variable `identifier` as it exists in a
 * single if- or else-block.
 */
struct Refinement {
  Refinement(std::string identifier, c10::TypePtr type)
      : identifier_(std::move(identifier)), type_(type) {}
  const std::string& identifier() const {
    return identifier_;
  }
  c10::TypePtr type() const {
    return type_;
  }

 private:
  std::string identifier_;
  c10::TypePtr type_;
};

using Refinements = std::vector<Refinement>;

/*
 * Structure for tracking type refinements on a conditional expression. Keeps
 * track of what refinements apply if the expression is true or false, and
 * offers logical conjuction operators to combine refinements from different
 * conditionals.
 */
struct RefinementSet {

  RefinementSet(Refinements true_refinements, Refinements false_refinements);
  RefinementSet(Refinement single);
  RefinementSet(Refinement single_true, Refinement single_false);
  RefinementSet() {} // empty

  RefinementSet And(const RefinementSet& rhs) const;
  RefinementSet Or(const RefinementSet& rhs) const;
  RefinementSet Not() const {
    return RefinementSet(false_refinements_, true_refinements_);
  }
  const std::vector<Refinement> activeRefinements() const {
    return true_refinements_;
  }

 private:
  static bool sameVar(const Refinement& a, const Refinement& b) {
    return a.identifier() == b.identifier();
  }
  static Refinements unionSet(const Refinements& a, const Refinements& b);
  static Refinements intersectSet(const Refinements& a, const Refinements& b);

  Refinements true_refinements_;
  Refinements false_refinements_;
};

} // namespace jit
} // namespace torch
