namespace torch {
namespace jit {

struct Refinement {
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(std::move(type)) {}
  const std::string& identifier() const {
    return identifier_;
  }
  TypePtr type() const {
    return type_;
  }

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
  RefinementSet And(const RefinementSet& rhs) const {
    // if the result of an AND is true, both a & b had to be true,
    // so we take the union of a.true_refinements and b.true_refinements.
    // if the result is false, either a or b could have been false,
    // so we take their intersection.
    return RefinementSet(
        unionSet(true_refinements_, rhs.true_refinements_),
        intersectSet(false_refinements_, rhs.false_refinements_));
  }
  RefinementSet Or(const RefinementSet& rhs) const {
    // if the result of an OR is true, either a & b could have been true,
    // so we take the intersection of a.true_refinements & b.true_refinements.
    // if the result is false, both a and b had to be false,
    // so we take their union.
    return RefinementSet(
        intersectSet(true_refinements_, rhs.true_refinements_),
        unionSet(false_refinements_, rhs.false_refinements_));
  }

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
  static Refinements unionSet(const Refinements& a, const Refinements& b) {
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
  static Refinements intersectSet(const Refinements& a, const Refinements& b) {
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
  Value* value() const {
    return value_;
  }
  const RefinementSet& refinements() const {
    return refinements_;
  }
  c10::optional<bool> staticIf() const {
    return static_if_;
  }

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
