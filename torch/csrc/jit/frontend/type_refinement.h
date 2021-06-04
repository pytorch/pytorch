#pragma once

#include <aten/src/ATen/core/jit_type.h>
#include <aten/src/ATen/core/jit_type_base.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct Refinement {
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(std::move(type)) {}

  const std::string& identifier() const;
  TypePtr type() const;

 private:
  std::string identifier_;
  TypePtr type_;
};

struct RefinementSet {
  // When a comparison like x is None is made, we associate type refinements
  // with its true value and its false value. If a boolean that has refinements
  // associated with it is used in a conditional of an if statement, the true
  // and false refinements are inserted into the corresponding blocks
  using Refinements = std::vector<Refinement>;

  RefinementSet(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)) {}
  RefinementSet(Refinement single) : RefinementSet({std::move(single)}, {}) {}
  RefinementSet(Refinement single_true, Refinement single_false)
      : RefinementSet(
            Refinements({std::move(single_true)}),
            Refinements({std::move(single_false)})) {}
  RefinementSet() = default; // empty

  RefinementSet And(const RefinementSet& rhs) const;
  RefinementSet Or(const RefinementSet& rhs) const;
  RefinementSet Not() const;
  const std::vector<Refinement> activeRefinements() const;

 private:
  static bool sameVar(const Refinement& a, const Refinement& b);
  static Refinements unionSet(const Refinements& a, const Refinements& b);
  static Refinements intersectSet(const Refinements& a, const Refinements& b);

  Refinements true_refinements_;
  Refinements false_refinements_;
};

struct CondValue {
  CondValue(
      Value* value,
      RefinementSet refinements,
      c10::optional<bool> static_if)
      : value_(value),
        refinements_(std::move(refinements)),
        static_if_(static_if) {}
  CondValue(
      Graph& g,
      const SourceRange& loc,
      bool static_value,
      RefinementSet refinements)
      : value_(g.insertConstant(static_value, loc)),
        refinements_(std::move(refinements)),
        static_if_(static_value) {}

  Value* value() const;
  const RefinementSet& refinements() const;
  c10::optional<bool> staticIf() const;

 private:
  Value* value_;
  RefinementSet refinements_;
  c10::optional<bool>
      static_if_; // certain expression cause us to emit a static if statement
                  // this value is present if this is the case.
                  // this is not equivalent to value_ being a constant
                  // it is possible for value_ to be constant but for
                  // the expression that produced it to not trigger the
                  // static if behavior. e.g. use of a variable assigned
                  // to a constant
};

RefinementSet findIsNoneRefinements(
    const Expr& lhs,
    Value* lhs_value,
    const Expr& rhs,
    Value* rhs_value,
    int tok);

} // namespace jit
} // namespace torch
