#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Simple recursive GCD.
template <typename T>
T gcd(T a, T b) {
  if (b == 0) {
    return a;
  }
  return gcd(b, a % b);
}

// Helper for determining if an Expr is a multi-lane primitive (e.g. Broadcast
// or Ramp).
bool isMultilanePrimitive(const Expr* e) {
  return dynamic_cast<const Broadcast*>(e) || dynamic_cast<const Ramp*>(e);
}

SimplifierHashType Term::hashVars() const {
  SimplifierHashType hash;
  for (auto* v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }

  return hash;
}

void Term::sort() {
  // order of ops important for float
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::sort(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
}

SimplifierHashType Polynomial::hashVars() const {
  SimplifierHashType hash;
  for (auto* v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }
  return hash;
}

void Polynomial::sort() {
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::sort(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
}

void MaxTerm::uniquefy() {
  std::sort(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));
}

void MinTerm::uniquefy() {
  std::sort(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](const Expr* a, const Expr* b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));
}

// Handles optimization cases for Broadcast/Ramp +/- Broadcast/Ramp
template <class Op>
const Expr* combineMultilane(const Expr* lhs, const Expr* rhs) {
  if (const Broadcast* bc = dynamic_cast<const Broadcast*>(lhs)) {
    if (const Broadcast* bcother = dynamic_cast<const Broadcast*>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret =
          new Broadcast(new Op(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (const Ramp* r = dynamic_cast<const Ramp*>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret =
          new Ramp(new Op(bc->value(), r->base()), r->stride(), r->lanes());
      return ret;
    }
  } else if (const Ramp* ramp = dynamic_cast<const Ramp*>(lhs)) {
    if (const Ramp* rother = dynamic_cast<const Ramp*>(rhs)) {
      if (ramp->lanes() != rother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret = new Ramp(
          new Op(ramp->base(), rother->base()),
          new Op(ramp->stride(), rother->stride()),
          ramp->lanes());
      return ret;
    }

    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }
      const Expr* ret = new Ramp(
          new Op(ramp->base(), bc->value()), ramp->stride(), ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

// Handles optimization cases for Broadcast/Ramp * Broadcast/Ramp
const Expr* mulMultilane(const Expr* lhs, const Expr* rhs) {
  if (const Broadcast* bc = dynamic_cast<const Broadcast*>(lhs)) {
    if (const Broadcast* bcother = dynamic_cast<const Broadcast*>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret =
          new Broadcast(new Mul(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (const Ramp* r = dynamic_cast<const Ramp*>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret = new Ramp(
          new Mul(bc->value(), r->base()),
          new Mul(bc->value(), r->stride()),
          r->lanes());
      return ret;
    }
  } else if (const Ramp* ramp = dynamic_cast<const Ramp*>(lhs)) {
    if (const Ramp* r = dynamic_cast<const Ramp*>(rhs)) {
      if (ramp->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret = new Ramp(
          new Mul(ramp->base(), r->base()),
          new Mul(ramp->stride(), r->stride()),
          r->lanes());
      return ret;
    }

    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      const Expr* ret = new Ramp(
          new Mul(bc->value(), ramp->base()),
          new Mul(bc->value(), ramp->stride()),
          ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

void PolynomialTransformer::addOrUpdateTerm(
    std::unordered_map<SimplifierHashType, const Term*>& varmap,
    const Term* term) {
  SimplifierHashType hash = term->hashVars();
  auto insertRes = varmap.emplace(hash, term);
  if (insertRes.second == false) {
    const Term* lt = insertRes.first->second;
    const Expr* termScalar = evaluateOp(new Add(lt->scalar(), term->scalar()));

    // If the term is canceled out, remove from the map.
    if (immediateEquals(termScalar, 0)) {
      varmap.erase(hash);
      return;
    }

    varmap[hash] = new Term(hasher_, termScalar, lt->variables());
  }
}

const Expr* PolynomialTransformer::addPolynomials(
    const Polynomial* lhs,
    const Polynomial* rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, const Term*> varmap;

  for (auto* lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }
  for (auto* rt : rhs->variables()) {
    addOrUpdateTerm(varmap, rt);
  }

  const Expr* newScalar = evaluateOp(new Add(lhs->scalar(), rhs->scalar()));
  return new Polynomial(hasher_, newScalar, varmap);
}

// Insert a new Term into the provided polynomial. If the new term has common
// variables to an existing term it is combined.
const Expr* PolynomialTransformer::insertTerm(
    const Polynomial* poly,
    const Term* term) {
  SimplifierHashType tHash = term->hashVars();
  std::vector<const Term*> newVars;

  bool found = false;
  for (auto* v : poly->variables()) {
    if (v->hashVars() == tHash) {
      const Expr* newScalar = evaluateOp(new Add(term->scalar(), v->scalar()));
      found = true;
      // Skip this term if we cancelled it out.
      if (immediateEquals(newScalar, 0)) {
        continue;
      }
      auto* term = new Term(hasher_, newScalar, v->variables());
      newVars.push_back(term);
    } else {
      newVars.push_back(v);
    }
  }

  if (!found) {
    newVars.push_back(term);
  }

  if (newVars.empty()) {
    return poly->scalar();
  }

  auto* Poly = new Polynomial(hasher_, poly->scalar(), newVars);
  return Poly;
}

const Expr* PolynomialTransformer::mutate(const Add* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    const Expr* result = evaluateOp(new Add(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto* ret = combineMultilane<Add>(lhs_new, rhs_new)) {
      return ret->accept_mutator(this);
    }
  }

  const Expr* scalar = nullptr;
  const Expr* variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = evaluateOp(lhs_new);
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = evaluateOp(rhs_new);
    variable = lhs_new;
  }

  // If there is a scalar, and it's zero: short circuit and return the other
  // side.
  if (scalar && immediateEquals(scalar, 0)) {
    auto* c = new Cast(v->dtype(), variable);
    return c->accept_mutator(this);
  }

  // If this is a floating point Add then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new Add(lhs_new, rhs_new);
  }

  const Polynomial* lhsPoly = dynamic_cast<const Polynomial*>(lhs_new);
  const Polynomial* rhsPoly = dynamic_cast<const Polynomial*>(rhs_new);

  if (lhsPoly && rhsPoly) {
    return addPolynomials(lhsPoly, rhsPoly);
  }

  const Term* lhsTerm = dynamic_cast<const Term*>(lhs_new);
  const Term* rhsTerm = dynamic_cast<const Term*>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return insertTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return insertTerm(rhsPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    // If the terms refer to the same variables: combine them.
    if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
      const Expr* newScalar =
          evaluateOp(new Add(lhsTerm->scalar(), rhsTerm->scalar()));

      // If the terms cancelled out, return zero.
      if (immediateEquals(newScalar, 0)) {
        return newScalar->accept_mutator(this);
      }

      return new Term(hasher_, newScalar, lhsTerm->variables());
    }

    // Otherwise this is a new polynomial with no scalar and two variable
    // terms.
    return new Polynomial(
        hasher_, getImmediateByType(v->dtype(), 0), lhsTerm, rhsTerm);
  }

  // Adds are commutative.
  const Polynomial* poly = lhsPoly ? lhsPoly : rhsPoly;

  // Add to Polynomial->scalar().
  if (scalar && poly) {
    const Expr* newScalar = evaluateOp(new Add(scalar, poly->scalar()));
    return new Polynomial(hasher_, newScalar, poly->variables());
  }

  // Simple Polynomial with a scalar and Term.
  const Term* term = lhsTerm ? lhsTerm : rhsTerm;
  if (scalar && term) {
    return new Polynomial(hasher_, scalar, term);
  }

  // Simple Term with a scalar and variable type.
  if (scalar) {
    return new Polynomial(
        hasher_,
        scalar,
        new Term(hasher_, getImmediateByType(v->dtype(), 1), variable));
  }

  // If LHS is neither Term not Polynomial, wrap it in a Term.
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = new Term(hasher_, getImmediateByType(v->dtype(), 1), lhs_new);
  }

  // Same for RHS.
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = new Term(hasher_, getImmediateByType(v->dtype(), 1), rhs_new);
  }

  // If we now have a poly and a term, we can insert.
  if (poly) {
    return insertTerm(poly, lhsTerm ? lhsTerm : rhsTerm);
  }

  if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
    return new Term(
        hasher_,
        evaluateOp(new Add(lhsTerm->scalar(), rhsTerm->scalar())),
        lhsTerm->variables());
  }

  // If all else fails we have a new Polynomial with two new variable Terms.
  return new Polynomial(
      hasher_, getImmediateByType(v->dtype(), 0), lhsTerm, rhsTerm);
}

const Expr* PolynomialTransformer::subTerms(
    const Term* lhs,
    const Term* rhs,
    bool negated) {
  // If RHS not already negated, negate it.
  if (!negated) {
    const Expr* minusOne = getImmediateByType(rhs->dtype(), -1);
    const Expr* negateScalar = evaluateOp(new Mul(minusOne, rhs->scalar()));
    rhs = new Term(hasher_, negateScalar, rhs->variables());
  }

  if (lhs->hashVars() == rhs->hashVars()) {
    const Expr* newScalar = evaluateOp(new Add(lhs->scalar(), rhs->scalar()));

    // If the terms cancel out, return zero.
    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }

    return new Term(hasher_, newScalar, lhs->variables());
  }

  return new Polynomial(
      hasher_,
      getImmediateByType(promoteTypes(lhs->dtype(), rhs->dtype()), 0),
      lhs,
      rhs);
}

// Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
// possible.
const Expr* PolynomialTransformer::subPolynomials(
    const Polynomial* lhs,
    const Polynomial* rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, const Term*> varmap;

  for (auto* lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }

  for (auto* rt : rhs->variables()) {
    // Polynomials add their terms, so negate the RHS's Terms.
    const Expr* negated =
        evaluateOp(new Mul(getImmediateByType(rt->dtype(), -1), rt->scalar()));
    Term* newRHS = new Term(hasher_, negated, rt->variables());
    addOrUpdateTerm(varmap, newRHS);
  }

  const Expr* newScalar = evaluateOp(new Sub(lhs->scalar(), rhs->scalar()));

  // No vars means this cancelled out to a scalar, return it unwrapped.
  if (varmap.empty()) {
    return newScalar;
  }

  // If there is no scalar and zero or one terms, don't wrap.
  if (immediateEquals(newScalar, 0)) {
    if (varmap.empty()) {
      return nullptr;
    }
    if (varmap.size() == 1) {
      return varmap.begin()->second;
    }
  }

  // Wrap new variables in a Polynomial.
  return new Polynomial(hasher_, newScalar, varmap);
}

const Expr* PolynomialTransformer::mutate(const Sub* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    const Expr* result = evaluateOp(new Sub(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto* ret = combineMultilane<Sub>(lhs_new, rhs_new)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);
    }
  }

  if (rhs_new->isConstant() && immediateEquals(rhs_new, 0)) {
    auto* c = new Cast(v->dtype(), lhs_new);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);
  }

  // If this is a floating point Sub then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new Sub(lhs_new, rhs_new);
  }

  const Polynomial* lhsPoly = dynamic_cast<const Polynomial*>(lhs_new);
  const Polynomial* rhsPoly = dynamic_cast<const Polynomial*>(rhs_new);

  if (lhsPoly && rhsPoly) {
    auto* ret = subPolynomials(lhsPoly, rhsPoly);
    if (!ret) {
      // Cancelled out completely.
      return getImmediateByType(v->dtype(), 0);
    }
    return ret;
  }

  const Term* lhsTerm = dynamic_cast<const Term*>(lhs_new);
  const Term* rhsTerm = dynamic_cast<const Term*>(rhs_new);

  // Polynomial - Term.
  if (lhsPoly && rhsTerm) {
    // Negate the term.
    const Expr* negate = evaluateOp(
        new Mul(getImmediateByType(rhsTerm->dtype(), -1), rhsTerm->scalar()));
    const Term* newTerm = new Term(hasher_, negate, rhsTerm->variables());
    return insertTerm(lhsPoly, newTerm);
  }

  // Term - Polynomial.
  if (rhsPoly && lhsTerm) {
    // Negate every part of the Polynomial.
    const Expr* minusOne = getImmediateByType(lhsTerm->dtype(), -1);
    const Expr* negateScalar = evaluateOp(new Mul(minusOne, rhsPoly->scalar()));

    std::vector<const Term*> variables;
    for (auto* t : rhsPoly->variables()) {
      const Expr* negate = evaluateOp(new Mul(minusOne, t->scalar()));
      variables.push_back(new Term(hasher_, negate, t->variables()));
    }

    Polynomial* newPoly = new Polynomial(hasher_, negateScalar, variables);
    return insertTerm(newPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    return subTerms(lhsTerm, rhsTerm, false);
  }

  bool lhsScalar = lhs_new->isConstant();
  bool rhsScalar = rhs_new->isConstant();

  if (lhsPoly && rhsScalar) {
    // Easy path, just sub the scalar component.
    const Expr* newScalar = evaluateOp(new Sub(lhsPoly->scalar(), rhs_new));
    return new Polynomial(hasher_, newScalar, lhsPoly->variables());
  }

  if (lhsScalar && rhsPoly) {
    // Sub the scalar component.
    const Expr* newScalar = evaluateOp(new Sub(lhs_new, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    const Expr* minusOne = getImmediateByType(rhsPoly->dtype(), -1);
    std::vector<const Term*> variables;
    for (auto* t : rhsPoly->variables()) {
      const Expr* negate = evaluateOp(new Mul(minusOne, t->scalar()));
      variables.push_back(new Term(hasher_, negate, t->variables()));
    }

    return new Polynomial(hasher_, newScalar, variables);
  }

  if (lhsTerm && rhsScalar) {
    // Negate the constant.
    const Expr* negate =
        evaluateOp(new Mul(getImmediateByType(rhs_new->dtype(), -1), rhs_new));
    return new Polynomial(hasher_, negate, lhsTerm);
  }

  if (lhsScalar && rhsTerm) {
    // Negate the RHS Term.
    const Expr* negate = evaluateOp(new Mul(
        getImmediateByType(rhsTerm->scalar()->dtype(), -1), rhsTerm->scalar()));

    return new Polynomial(
        hasher_, lhs_new, new Term(hasher_, negate, rhsTerm->variables()));
  }

  // simple term with a scalar and variable type.
  if (lhsScalar) {
    // Create a negated term.
    return new Polynomial(
        hasher_,
        lhs_new,
        new Term(hasher_, getImmediateByType(v->dtype(), -1), rhs_new));
  }

  if (rhsScalar) {
    // Negate the scalar.
    const Expr* negate =
        evaluateOp(new Mul(getImmediateByType(rhs_new->dtype(), -1), rhs_new));
    return new Polynomial(
        hasher_,
        negate,
        new Term(hasher_, getImmediateByType(v->dtype(), 1), lhs_new));
  }

  // no scalar...
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = new Term(hasher_, getImmediateByType(v->dtype(), 1), lhs_new);
  }

  bool createdRHSnegated = false;
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = new Term(hasher_, getImmediateByType(v->dtype(), -1), rhs_new);
    createdRHSnegated = true;
  }

  if (lhsTerm && rhsTerm) {
    return subTerms(lhsTerm, rhsTerm, createdRHSnegated);
  }

  // Insert wrapped Term into LHS Polynomial.
  if (lhsPoly) {
    CHECK(rhsTerm);
    return insertTerm(lhsPoly, rhsTerm);
  }

  // Insert wrapper Term into negated RHS Poly.
  if (rhsPoly) {
    CHECK(lhsTerm);
    const Expr* minusOne = getImmediateByType(rhsPoly->dtype(), -1);
    const Expr* newScalar = evaluateOp(new Mul(minusOne, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    std::vector<const Term*> variables;
    for (auto* t : rhsPoly->variables()) {
      const Expr* negate = evaluateOp(new Mul(minusOne, t->scalar()));
      variables.push_back(new Term(hasher_, negate, t->variables()));
    }

    auto* poly = new Polynomial(hasher_, newScalar, variables);
    return insertTerm(poly, lhsTerm);
  }

  return new Polynomial(
      hasher_, getImmediateByType(v->dtype(), 0), lhsTerm, rhsTerm);
}

// Multiply two terms together, usually creating a new term with the variable
// lists concatenated.
const Term* PolynomialTransformer::mulTerms(const Term* lhs, const Term* rhs) {
  const Expr* scalar = evaluateOp(new Mul(lhs->scalar(), rhs->scalar()));
  if (immediateEquals(scalar, 0)) {
    return nullptr;
  }

  // Can reorder here since floating point ops don't get put into Terms.
  std::vector<const Expr*> variables;
  std::vector<const Expr*> multilaneVariables;
  // For now don't handle exponents.
  for (auto* c : lhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }
  for (auto* c : rhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }

  // Merge all the multilane vars:
  const Expr* lastNode{nullptr};
  for (auto* node : multilaneVariables) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      if (auto* next = mulMultilane(lastNode, node)) {
        lastNode = next->accept_mutator(this);
      } else {
        variables.push_back(lastNode);
        lastNode = node;
      }
    }
  }
  if (lastNode) {
    variables.push_back(lastNode);
  }

  return new Term(hasher_, scalar, variables);
}

// Multiply a Polynomial by a Term.
const Expr* PolynomialTransformer::polyByTerm(
    const Polynomial* poly,
    const Term* term) {
  std::vector<const Term*> newTerms;

  // scalar Term
  const Expr* scalar = evaluateOp(new Mul(poly->scalar(), term->scalar()));

  for (auto* var : poly->variables()) {
    const Term* newTerm = mulTerms(var, term);
    if (newTerm) {
      newTerms.push_back(newTerm);
    }
  }

  if (newTerms.empty()) {
    return scalar;
  }

  return new Polynomial(hasher_, scalar, std::move(newTerms));
}

// Does multiplying these two expressions make a Rounding Off operation.
// e.g. LHS = (x/y),  RHS = y => (x / y) * y => RoundOff(x, y).
const Expr* PolynomialTransformer::isRoundOff(
    const Expr* lhs,
    const Expr* rhs) {
  const Div* div{nullptr};
  const Expr* other{nullptr};

  if ((div = dynamic_cast<const Div*>(lhs))) {
    other = rhs;
  } else if ((div = dynamic_cast<const Div*>(rhs))) {
    other = lhs;
  } else {
    return nullptr;
  }

  const Expr* denom = div->rhs();

  if (const Term* denomTerm = dynamic_cast<const Term*>(denom)) {
    if (immediateEquals(denomTerm->scalar(), 1) &&
        denomTerm->variables().size() == 1) {
      denom = denomTerm->variables()[0];
    }
  }

  if (hasher_.hash(denom) == hasher_.hash(other)) {
    // If the denominator is equal to the other, then yes it's a RoundOff.
    return new RoundOff(div->lhs(), div->rhs());
  }

  if (denom->isConstant() && other->isConstant()) {
    if (immediateEquals(denom, 0) || immediateEquals(other, 0)) {
      return nullptr;
    }
    // If they are both scalar we may be able to find a common factor.
    if (immediateEquals(evaluateOp(new Mod(other, denom)), 0)) {
      Expr* scalar = evaluateOp(new Div(other, denom));
      Expr* newDenom = evaluateOp(new Div(other, scalar));
      return new Term(hasher_, scalar, new RoundOff(div->lhs(), newDenom));
    }
  }

  return nullptr;
}

// Inserts a new component into a term, looking for opportunities to simplify.
const Expr* PolynomialTransformer::insertIntoTerm(
    const Term* term,
    const Expr* expr) {
  std::vector<const Expr*> vars;

  // Search for RoundOffs.
  bool merged{false};
  for (auto* component : term->variables()) {
    if (auto* roundoff = isRoundOff(component, expr)) {
      vars.push_back(roundoff);
      merged = true;
    } else {
      vars.push_back(component);
    }
  }

  if (!merged) {
    vars.push_back(expr);
  }

  if (vars.size() == 1 && immediateEquals(term->scalar(), 1)) {
    return vars[0];
  }

  return new Term(hasher_, term->scalar(), vars);
}

const Expr* PolynomialTransformer::mutate(const Mul* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(new Mul(lhs_new, rhs_new));
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto* ret = mulMultilane(lhs_new, rhs_new)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);
    }
  }

  // Order doesn't matter.
  const Expr* scalar = nullptr;
  const Expr* variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = lhs_new;
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = rhs_new;
    variable = lhs_new;
  }

  // Handle special case mul by 1 since thats safe for floating point, even if
  // it's Nan/Inf.
  if (scalar && immediateEquals(scalar, 1)) {
    auto* c = new Cast(v->dtype(), variable);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);
  }

  // If this is a floating point Mul then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new Mul(lhs_new, rhs_new);
  }

  // Handle special case mul by 0.
  if (scalar && immediateEquals(scalar, 0)) {
    return getImmediateByType(v->dtype(), 0);
  }

  // Catch cases of rounding (Div(A/B) * B).
  if (auto* ret = isRoundOff(lhs_new, rhs_new)) {
    return ret;
  } else if (auto* ret = isRoundOff(v->lhs(), v->rhs())) {
    // We can break the Round + Mod pattern via factorization of the Div, so
    // check whether it would have worked on the unsimplified tree. If so, we
    // need to simplify again.
    return ret->accept_mutator(this);
  }

  const Polynomial* lhsPoly = dynamic_cast<const Polynomial*>(lhs_new);
  const Polynomial* rhsPoly = dynamic_cast<const Polynomial*>(rhs_new);

  if (lhsPoly && rhsPoly) {
    // This expands to more terms that we can't generally fix without variable
    // factorization, it's more efficient to just leave these as Muls.
    return new Mul(lhsPoly, rhsPoly);
  }

  const Term* lhsTerm = dynamic_cast<const Term*>(lhs_new);
  const Term* rhsTerm = dynamic_cast<const Term*>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return polyByTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return polyByTerm(rhsPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    return mulTerms(lhsTerm, rhsTerm);
  }

  if (scalar && lhsTerm) {
    const Expr* newScalar = evaluateOp(new Mul(scalar, lhsTerm->scalar()));
    return new Term(hasher_, newScalar, lhsTerm->variables());
  }

  if (scalar && rhsTerm) {
    const Expr* newScalar = evaluateOp(new Mul(scalar, rhsTerm->scalar()));
    return new Term(hasher_, newScalar, rhsTerm->variables());
  }

  // If this is a scalar * a Polynomial, push the scalar term down.
  // We can wrap the scalar with a Term and use polyByTerm.
  if (scalar && lhsPoly) {
    return polyByTerm(lhsPoly, new Term(hasher_, scalar));
  }
  if (scalar && rhsPoly) {
    return polyByTerm(rhsPoly, new Term(hasher_, scalar));
  }

  // simple term with a scalar and variable type.
  if (scalar) {
    return new Term(hasher_, scalar, variable);
  }

  // Multiplying Polynomial by variable can be wrapped in a term and handled
  // by polyByTerm also.
  if (lhsPoly) {
    auto* term =
        new Term(hasher_, getImmediateByType(rhs_new->dtype(), 1), rhs_new);
    return polyByTerm(lhsPoly, term);
  }
  if (rhsPoly) {
    auto* term =
        new Term(hasher_, getImmediateByType(lhs_new->dtype(), 1), lhs_new);
    return polyByTerm(rhsPoly, term);
  }

  // Multiplying Term by a variable is equivalent to adding the variable to
  // the term's list of vars.
  if (lhsTerm) {
    return insertIntoTerm(lhsTerm, rhs_new);
  }
  if (rhsTerm) {
    return insertIntoTerm(rhsTerm, lhs_new);
  }

  // Two variables, create a new Term.
  return new Term(hasher_, getImmediateByType(v->dtype(), 1), lhs_new, rhs_new);
}

const Expr* factorizeDivision(const Expr* lhs_new, const Expr* rhs_new) {
  if (!lhs_new || !rhs_new) {
    return nullptr;
  }

  const Expr* leftScalar = lhs_new->isConstant() ? lhs_new : nullptr;
  const Expr* rightScalar = rhs_new->isConstant() ? rhs_new : nullptr;

  auto* lhsTerm = dynamic_cast<const Term*>(lhs_new);
  auto* rhsTerm = dynamic_cast<const Term*>(rhs_new);
  if (lhsTerm) {
    leftScalar = lhsTerm->scalar();
  }

  if (rhsTerm) {
    rightScalar = rhsTerm->scalar();
  }

  if (!leftScalar || !rightScalar) {
    return nullptr;
  }

  long left = immediateAs<long>(leftScalar);
  long right = immediateAs<long>(rightScalar);

  long GCD = gcd<long>(left, right);
  if (GCD <= 1) {
    return nullptr;
  }

  leftScalar = evaluateOp(
      new Div(leftScalar, getImmediateByType(leftScalar->dtype(), GCD)));
  rightScalar = evaluateOp(
      new Div(rightScalar, getImmediateByType(rightScalar->dtype(), GCD)));

  if (lhsTerm) {
    lhs_new = new Term(lhsTerm->hasher(), leftScalar, lhsTerm->variables());
  } else {
    lhs_new = leftScalar;
  }

  if (rhsTerm) {
    rhs_new = new Term(rhsTerm->hasher(), rightScalar, rhsTerm->variables());
  } else {
    rhs_new = rightScalar;
  }

  return new Div(lhs_new, rhs_new);
}

const Expr* PolynomialTransformer::mutate(const Div* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(new Div(lhs_new, rhs_new));
  }

  // If this is a floating point Div then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new Div(lhs_new, rhs_new);
  }

  // If the numerator is zero, so is the result.
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return lhs_new;
  }

  // If the denominator is one, return numerator.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    return lhs_new;
  }

  // If numberator and denominator are equal the result is 1.
  // Unless the demoninator could be zero.
  // if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
  //   return getImmediateByType(v->dtype(), 1);
  // }

  if (auto ret = factorizeDivision(lhs_new, rhs_new)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return ret->accept_mutator(this);
  }

  return new Div(lhs_new, rhs_new);
}

const Expr* PolynomialTransformer::mutate(const Mod* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(new Mod(lhs_new, rhs_new));
  }

  // 0 % x => 0.
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return lhs_new;
  }

  // x % 1 == 0.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return getImmediateByType(v->dtype(), 0);
  }

  // x % x => 0.
  if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
    return getImmediateByType(v->dtype(), 0);
  }

  const Term* lhsTerm = dynamic_cast<const Term*>(lhs_new);
  if (!lhsTerm) {
    const Polynomial* lhsPoly = dynamic_cast<const Polynomial*>(lhs_new);
    if (lhsPoly) {
      // Can still optimize this out if we can factorize the polynomial.
      lhsTerm = factorizePolynomial(lhsPoly);
    }
  }

  if (lhsTerm) {
    // ((C1 * C2) * x) % C1 => 0.
    if (rhs_new->isConstant() &&
        immediateEquals(evaluateOp(new Mod(lhsTerm->scalar(), rhs_new)), 0)) {
      return getImmediateByType(v->dtype(), 0);
    }

    // (x * y * z) % x => 0.
    for (auto* component : lhsTerm->variables()) {
      if (hasher_.hash(component) == hasher_.hash(rhs_new)) {
        return getImmediateByType(v->dtype(), 0);
      }
    }

    // (6 * x * y) % (3 * x * y) => 0.
    // also, (x * y * z) % (z * y) => 0.
    // This requires all variable terms found in the RHS to be present in the
    // LHS.
    const Term* rhsTerm = dynamic_cast<const Term*>(rhs_new);
    if (rhsTerm) {
      auto& lVars = lhsTerm->variables();
      auto& rVars = rhsTerm->variables();
      size_t rLeft = rVars.size();

      auto rIt = rVars.begin();

      for (auto lIt = lVars.begin(); lIt != lVars.end() && !rVars.empty();
           ++lIt) {
        auto lHash = hasher_.hash(*lIt);
        for (; rIt != rVars.end(); ++rIt) {
          auto rHash = hasher_.hash(*rIt);
          if (lHash == rHash) {
            --rLeft;
            break;
          } else if (lHash < rHash) {
            break;
          }
        }
      }

      if (rLeft == 0 &&
          immediateEquals(
              evaluateOp(new Mod(lhsTerm->scalar(), rhsTerm->scalar())), 0)) {
        return getImmediateByType(v->dtype(), 0);
      }
    }
  }

  return new Mod(lhs_new, rhs_new);
}

namespace {

// Combines two MinTerm / MaxTerm expressions into one.
// The first type on the template refers to the op, as in Min or Max and the
// second type refers to the corresponding term, as in MinTerm or MaxTerm.
template <class Op, class OpTerm>
const Expr* combineMinMaxTerms(
    const Expr* lhs,
    const Expr* rhs,
    bool propagate_nans,
    HashProvider& hasher) {
  auto combine_scalars = [&](const Expr* c1, const Expr* c2) -> const Expr* {
    if (c1 && c2) {
      return evaluateOp(new Op(c1, c2, propagate_nans));
    }
    if (c1) {
      return c1;
    }
    return c2;
  };

  auto combine_opterms = [&](const OpTerm* m1, const OpTerm* m2) {
    const Expr* scalar = combine_scalars(m1->scalar(), m2->scalar());
    std::vector<const Expr*> variables;
    for (auto v : m1->variables()) {
      variables.push_back(v);
    }
    for (auto v : m2->variables()) {
      variables.push_back(v);
    }
    return new OpTerm(hasher, scalar, propagate_nans, std::move(variables));
  };

  auto add_expr_to_opterm = [&](const Expr* expr, const OpTerm* opterm) {
    const Expr* scalar = nullptr;
    std::vector<const Expr*> variables;
    if (opterm) {
      scalar = opterm->scalar();
      variables = opterm->variables();
    }
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (expr->isConstant()) {
      scalar = combine_scalars(scalar, expr);
    } else {
      variables.push_back(expr);
    }
    return new OpTerm(hasher, scalar, propagate_nans, std::move(variables));
  };

  const OpTerm* lhs_opterm = dynamic_cast<const OpTerm*>(lhs);
  const OpTerm* rhs_opterm = dynamic_cast<const OpTerm*>(rhs);
  if (lhs_opterm && lhs_opterm->propagate_nans() != propagate_nans) {
    return new Op(lhs, rhs, propagate_nans);
  }
  if (rhs_opterm && rhs_opterm->propagate_nans() != propagate_nans) {
    return new Op(lhs, rhs, propagate_nans);
  }

  if (lhs_opterm && rhs_opterm) {
    return combine_opterms(lhs_opterm, rhs_opterm);
  } else if (lhs_opterm) {
    return add_expr_to_opterm(rhs, lhs_opterm);
  } else if (rhs_opterm) {
    return add_expr_to_opterm(lhs, rhs_opterm);
  }
  return add_expr_to_opterm(rhs, add_expr_to_opterm(lhs, nullptr));
}

// Returns true if op is one of the 2 operands in opterm and also returns
// the other op of opterm in other_op.
template <class OpTerm>
bool isOperandInMinMaxTerm(
    const OpTerm* opterm,
    const Expr* op,
    HashProvider& hasher,
    const Expr** other_op) {
  if (opterm->variables().size() != 2) {
    return false;
  }
  auto lhs = opterm->variables()[0];
  auto rhs = opterm->variables()[1];
  auto op_hash = hasher.hash(op);
  if (hasher.hash(lhs) == op_hash) {
    *other_op = rhs;
    return true;
  } else if (hasher.hash(rhs) == op_hash) {
    *other_op = lhs;
    return true;
  }
  return false;
};

// Simplifies the nested min-max pattern like:
//   * Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
//   * Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
// This function is called while processing the outer Min / Max ops.
// At that point the inner Min / Max ops would have been converted to
// MinTerm / MaxTerm as appropriate. So, this function checks for those
// term expressions in the given lhs and rhs.
//
// The first type of the template must be the term type corresponding to the
// outer op (e.g. MaxTerm) and the second type of the template must be the term
// type corresponding to the expected inner op (e.g. MinTerm).
template <class OpTerm, class OtherOpTerm>
bool simplifyNestedMinMax(
    const Expr* lhs,
    const Expr* rhs,
    bool propagate_nans,
    HashProvider& hasher,
    const Expr** new_op) {
  auto lhs_opterm = dynamic_cast<const OtherOpTerm*>(lhs);
  auto rhs_opterm = dynamic_cast<const OtherOpTerm*>(rhs);
  if (lhs_opterm && rhs_opterm &&
      lhs_opterm->propagate_nans() == propagate_nans &&
      rhs_opterm->propagate_nans() == propagate_nans) {
    if (!lhs_opterm->scalar() && !rhs_opterm->scalar()) {
      if (lhs_opterm->variables().size() == 2 &&
          rhs_opterm->variables().size() == 2) {
        auto rhs_v1 = rhs_opterm->variables()[0];
        auto rhs_v2 = rhs_opterm->variables()[1];
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        const Expr* new_op_lhs;
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v1, hasher, &new_op_lhs)) {
          auto inner_op =
              new OpTerm(hasher, nullptr, propagate_nans, new_op_lhs, rhs_v2);
          *new_op = new OtherOpTerm(
              hasher, nullptr, propagate_nans, rhs_v1, inner_op);
          return true;
        }
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v2, hasher, &new_op_lhs)) {
          auto inner_op =
              new OpTerm(hasher, nullptr, propagate_nans, new_op_lhs, rhs_v1);
          *new_op = new OtherOpTerm(
              hasher, nullptr, propagate_nans, rhs_v2, inner_op);
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

const Expr* PolynomialTransformer::mutate(const Max* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(new Max(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  const Expr* diff = new Sub(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) > 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const Expr* new_op;
  if (simplifyNestedMinMax<MaxTerm, MinTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Max, MaxTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

const Expr* PolynomialTransformer::mutate(const Min* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(new Min(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  const Expr* diff = new Sub(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) < 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const Expr* new_op;
  if (simplifyNestedMinMax<MinTerm, MaxTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Min, MinTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

const Expr* PolynomialTransformer::mutate(const CompareSelect* v) {
  const Expr* lhs_new = v->lhs()->accept_mutator(this);
  const Expr* rhs_new = v->rhs()->accept_mutator(this);
  const Expr* true_branch = v->ret_val1()->accept_mutator(this);
  const Expr* false_branch = v->ret_val2()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    const Expr* v_new = new CompareSelect(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
    return evaluateOp(v_new);
  }

  // If the comparison is done in float, don't attempt diff simplification,
  // since we can't correctly handle NaN.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new CompareSelect(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
  }

  // If diff is constant, we can determine it.
  const Expr* diff = new Sub(rhs_new, lhs_new);
  diff = diff->accept_mutator(this);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  if (!diff->isConstant()) {
    return new CompareSelect(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        v->compare_select_op(),
        v->bias());
  }

  bool equal = immediateEquals(diff, 0);
  bool lhsSmaller = !equal && !immediateIsNegative(diff);

  switch (v->compare_select_op()) {
    case CompareSelectOperation::kEQ:
      return equal ? true_branch : false_branch;
    case CompareSelectOperation::kGT:
      return (lhsSmaller || equal) ? false_branch : true_branch;
    case CompareSelectOperation::kGE:
      return lhsSmaller ? false_branch : true_branch;
    case CompareSelectOperation::kLT:
      return lhsSmaller ? true_branch : false_branch;
    case CompareSelectOperation::kLE:
      return (lhsSmaller || equal) ? true_branch : false_branch;
    case CompareSelectOperation::kNE:
      return equal ? false_branch : true_branch;
  }

  // should not be possible but just in case.
  return new CompareSelect(
      lhs_new,
      rhs_new,
      true_branch,
      false_branch,
      v->compare_select_op(),
      v->bias());
}

const Expr* PolynomialTransformer::mutate(const Intrinsics* v) {
  std::vector<const Expr*> new_params;
  bool changed = false;
  bool allConstant = true;
  for (const auto* p : v->params()) {
    const Expr* new_child = p->accept_mutator(this);
    new_params.push_back(new_child);

    changed |= p != new_child;
    allConstant &= new_child->isConstant();
  }

  const Expr* node = v;
  if (changed) {
    node = new Intrinsics(v->op_type(), new_params);
  }

  if (!allConstant || !v->isPure()) {
    return node;
  }

  // we're evaluating, but the evaluator only supports float intrinsics.
  std::vector<const Expr*> const_params;
  changed = false;
  for (const auto* p : new_params) {
    if (p->dtype().scalar_type() == ScalarType::Float) {
      const_params.push_back(p);
    } else {
      const_params.push_back(
          new Cast(Dtype(ScalarType::Float, p->dtype().lanes()), p));
      changed = true;
    }
  }

  if (changed) {
    node = new Intrinsics(v->op_type(), const_params);
  }
  return evaluateOp(node);
}

const Expr* PolynomialTransformer::mutate(const Cast* v) {
  const Expr* node = v->src_value()->accept_mutator(this);
  if (node->isConstant()) {
    return evaluateOp(new Cast(v->dtype(), node));
  }

  if (v->dtype() == node->dtype()) {
    return node;
  }

  return new Cast(v->dtype(), node);
}

const Expr* PolynomialTransformer::mutate(const IfThenElse* v) {
  const Expr* condition = v->condition();
  const Expr* true_value = v->true_value();
  const Expr* false_value = v->false_value();
  const Expr* condition_new = condition->accept_mutator(this);
  const Expr* true_value_new = true_value->accept_mutator(this);
  const Expr* false_value_new = false_value->accept_mutator(this);

  // If the condition is constant then we can choose the right branch now.
  if (condition_new->isConstant()) {
    if (!immediateEquals(condition_new, 0)) {
      return true_value_new;
    } else {
      return false_value_new;
    }
  }

  // If both branches are the same then don't do the condition.
  if (hasher_.hash(true_value_new) == hasher_.hash(false_value_new)) {
    return true_value_new;
  }

  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return new IfThenElse(condition_new, true_value_new, false_value_new);
}

Stmt* IRSimplifierBase::mutate(const Cond* v) {
  const Expr* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  const Expr* cond_new = cond_old->accept_mutator(this);
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

  // If the condition is constant then we can choose the right branch now.
  if (cond_new->isConstant()) {
    if (!immediateEquals(cond_new, 0)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return true_new ? Stmt::clone(true_new) : nullptr;
    } else {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return false_new ? Stmt::clone(false_new) : nullptr;
    }
  }

  // If both branches are the same then don't do the condition.
  if (true_new && false_new &&
      hasher_.hash(true_new) == hasher_.hash(false_new)) {
    return Stmt::clone(true_new);
  }

  Block* true_block = dynamic_cast<Block*>(true_new);
  Block* false_block = dynamic_cast<Block*>(false_new);
  bool true_empty = !true_new || (true_block && true_block->nstmts() == 0);
  bool false_empty = !false_new || (false_block && false_block->nstmts() == 0);

  if (true_empty && false_empty) {
    return new Block({});
  }

  if (cond_old == cond_new && true_old == true_new && false_old == false_new) {
    return (Stmt*)v;
  }

  if (true_old && true_new == true_old) {
    true_new = Stmt::clone(true_old);
  }
  if (false_old && false_new == false_old) {
    false_new = Stmt::clone(false_old);
  }

  return new Cond(cond_new, true_new, false_new);
}

Stmt* handleForCondReordering(const For* loop, Cond* cond) {
  if (cond->false_stmt()) {
    return nullptr;
  }

  auto condition_vars = VarFinder::find(cond->condition());
  for (auto* v : condition_vars) {
    // If the condition depends on a Var that is modified in the loop body, it
    // may not be safe to reorder.
    if (ModifiesVarChecker::check(loop, v)) {
      return nullptr;
    }
  }

  For* new_f = loop->cloneWithNewBody(Stmt::clone(cond->true_stmt()));
  return cond->cloneWithNewBody(new_f);
}

Stmt* IRSimplifierBase::mutate(const For* v) {
  const Expr* var = v->var();
  const Expr* start = v->start();
  const Expr* stop = v->stop();
  Stmt* body = v->body();
  LoopOptions loop_options = v->loop_options();
  const Expr* var_new_expr = var->accept_mutator(this);
  const Var* var_new = dynamic_cast<const Var*>(var_new_expr);
  const Expr* start_new = start->accept_mutator(this);
  const Expr* stop_new = stop->accept_mutator(this);
  Stmt* body_new = body;

  const Expr* loops = new Sub(stop_new, start_new);
  loops = loops->accept_mutator(this);
  if (loop_options.isDefault() && loops->isConstant()) {
    if (immediateEquals(loops, 0)) {
      return new Block({});
    } else if (immediateEquals(loops, 1)) {
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);
      return body_new;
    }
  }

  body_new = body_new->accept_mutator(this);
  if (!body_new) {
    return new Block({});
  }

  if (auto* block = dynamic_cast<Block*>(body_new)) {
    if (block->nstmts() == 0) {
      return new Block({});
    }

    if (block->nstmts() == 1) {
      if (auto* cond = dynamic_cast<Cond*>(block->front())) {
        Stmt* reordered = handleForCondReordering(v, cond);
        if (reordered) {
          return reordered->accept_mutator(this);
        }
      }
    }
  }

  if (var == var_new && start == start_new && stop == stop_new &&
      body == body_new) {
    return (Stmt*)v;
  }
  if (body_new == body) {
    body_new = Stmt::clone(body);
  }
  return new For(var_new, start_new, stop_new, body_new, loop_options);
}

Stmt* IRSimplifierBase::mutate(const Block* v) {
  std::vector<Stmt*> stmts;
  // Flatten sub-blocks:
  for (Stmt* stmt : *v) {
    Stmt* stmt_new = stmt->accept_mutator(this);
    if (stmt_new == nullptr) {
      continue;
    }

    if (auto* subBlock = dynamic_cast<Block*>(stmt_new)) {
      for (Block::iterator I = subBlock->begin(), E = subBlock->end();
           I != E;) {
        // Be careful to avoid invalidating the iterator.
        Stmt* s = *(I++);
        subBlock->remove_stmt(s);
        stmts.push_back(s);
      }
    } else {
      stmts.push_back(Stmt::clone(stmt_new));
    }
  }

  return new Block(stmts);
}

// TermExpander

const Expr* TermExpander::mutate(const Term* v) {
  const Expr* newScalar = v->scalar()->accept_mutator(this);
  if (immediateEquals(newScalar, 0)) {
    return newScalar;
  }

  std::vector<const Expr*> vars;
  std::vector<const Expr*> multilaneVars;

  // Assume we can reorder here because we wont merge floating terms.
  const Expr* lastNode{nullptr};
  for (auto* var : v->variables()) {
    const Expr* node = var->accept_mutator(this);
    if (const Mul* mul = dynamic_cast<const Mul*>(node)) {
      // If the sub-Expr resolved to a multiplication, lift it into this
      // term.
      if (isMultilanePrimitive(mul->lhs())) {
        multilaneVars.push_back(mul->lhs());
      } else {
        vars.push_back(mul->lhs());
      }

      if (isMultilanePrimitive(mul->rhs())) {
        multilaneVars.push_back(mul->rhs());
      } else {
        vars.push_back(mul->rhs());
      }
    } else {
      if (isMultilanePrimitive(node)) {
        multilaneVars.push_back(node);
      } else {
        vars.push_back(node);
      }
    }
  }

  for (auto* node : multilaneVars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      lastNode = mulMultilane(lastNode, node);
      // simplify first, then re-expand.
      lastNode = lastNode->accept_mutator(simplifier_);
      lastNode = lastNode->accept_mutator(this);
    }
  }

  for (auto* node : vars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      lastNode = new Mul(lastNode, node);
    }
  }

  if (!immediateEquals(newScalar, 1)) {
    if (lastNode) {
      // We want to avoid a leaving a CastNode on the scalar, so handle that
      // now.
      auto termDtype = v->scalar()->dtype();
      auto lastNodeDtype = lastNode->dtype();
      if (termDtype != lastNodeDtype) {
        const Expr* castV = v->scalar();
        // Take care of lane mismatch first.
        if (termDtype.lanes() != lastNodeDtype.lanes()) {
          castV = new Broadcast(v->scalar(), lastNodeDtype.lanes());
        }
        // Now take care of scalar type as well.
        if (termDtype.scalar_type() != lastNodeDtype.scalar_type()) {
          castV = new Cast(lastNode->dtype(), castV);
          // For scalars, we can simplify the cast further.
          if (lastNodeDtype.lanes() == 1) {
            castV = evaluateOp(castV);
          }
        }
        lastNode = new Mul(castV, lastNode);
      } else {
        lastNode = new Mul(v->scalar(), lastNode);
      }
    } else {
      lastNode = v->scalar();
    }
  }

  return lastNode;
}

// Returns an immediate containing the greatest common divisor of all terms
// (inc. the scalar term) in the polynomial. If the GCD is uninteresting
// (e.g. 1) then returns nullptr.
const Expr* polyGCD(const Polynomial* poly) {
  const Expr* scalar = poly->scalar();
  const std::vector<const Term*>& variables = poly->variables();

  // We ony want to factorize if we're saving complete operations, i.e. no
  // value in factorizing 6x + 4y into 2 * (3x + 2y) since we don't save work.
  int opsSaved = 1; // default to saving the scalar.
  long GCD = std::abs(immediateAs<long>(scalar));
  for (auto* t : variables) {
    long termScalar = std::abs(immediateAs<long>(t->scalar()));
    long newGCD = gcd(std::max(GCD, termScalar), std::min(GCD, termScalar));
    if (newGCD == 1) {
      return nullptr;
    }

    if (GCD != newGCD) {
      opsSaved = 0;
      GCD = newGCD;
    }

    if (GCD == termScalar) {
      opsSaved++;
    }
  }

  if (opsSaved == 0) {
    return nullptr;
  }

  if (GCD == 0) {
    return nullptr;
  }

  // Not worth, can be a Sub.
  if (GCD == -1 && opsSaved == 1) {
    return nullptr;
  }

  return getImmediateByType(poly->dtype(), GCD);
}

// A ModRound is a div-mod-mul in which the divisor in div and multiplier in mul
// are identical and not equal to 1.
// In a ModRound x/y%z*y*c (c is constant), 'scalar' denotes c, 'denominator'
// denotes x, 'divisor' denotes y and 'mod_divisor' denotes z.
class ModRound {
 public:
  ModRound(
      const Expr* scalar,
      const Expr* denom,
      const Expr* divisor,
      const Expr* mod_divisor)
      : scalar(scalar),
        denom(denom),
        divisor(divisor),
        mod_divisor(mod_divisor) {}
  const Expr* scalar;
  const Expr* denom;
  const Expr* divisor;
  const Expr* mod_divisor;
};

c10::optional<class ModRound*> isModRound(const Term* e) {
  const Div* div{nullptr};
  const Mod* mod{nullptr};
  const Expr* denom{nullptr};
  const Expr* divisor{nullptr};
  const Expr* mod_divisor{nullptr};
  const Expr* multiplier = e->scalar();
  const Expr* scalar{nullptr};
  const Expr* other{nullptr};

  for (auto* m : e->variables()) {
    if (m->expr_type() == IRNodeType::kMod) {
      // TODO: currently only identify terms with one variable being mod; it is
      // possible to extend this if we have to handle terms like (t/(x%2 * y) %
      // z) * (x%2 *y).
      if (!mod) {
        mod = dynamic_cast<const Mod*>(m);
      } else {
        return c10::nullopt;
      }
    } else {
      // Take care of special cases before multiplying the scalar and variable.
      if (multiplier->isConstant()) {
        // Take care of lane mismatch first.
        if (multiplier->dtype().lanes() != m->dtype().lanes()) {
          multiplier = new Broadcast(multiplier, m->dtype().lanes());
        }
        // Take care of scalar type mismatch.
        if (multiplier->dtype().scalar_type() != m->dtype().scalar_type()) {
          multiplier = new Cast(m->dtype(), multiplier);
          if (m->dtype().lanes() == 1) {
            multiplier = evaluateOp(multiplier);
          }
        }
      }

      // All non-mod vairables are considered as part of the multiplier.
      multiplier = new Mul(multiplier, m);
    }
  }
  multiplier = IRSimplifier::simplify(multiplier);

  if (!mod) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c10::nullopt;
  }

  mod_divisor = IRSimplifier::simplify(mod->rhs());
  other = mod->lhs();

  if (!(div = dynamic_cast<const Div*>(other))) {
    return c10::nullopt;
  }

  divisor = IRSimplifier::simplify(div->rhs());
  other = div->lhs();

  denom = IRSimplifier::simplify(other);

  // Deny cases in which divisor!=multiplier.
  HashProvider& hasher = e->hasher();
  if (hasher.hash(divisor) != hasher.hash(multiplier)) {
    // TODO: currently we do not extract a common factor if divisor and
    // multiplier are not constants. The extraction is not supported (e.g.,
    // x*2/x -> 2) in IRSimplifier.simplify because x could be 0. As future
    // work, we can extend division to 2 versions: 1) division for customers
    // that has to be strictly simplified and 2) division we introduced in our
    // transformations which can be simplified without considering 0s, e.g.,
    // Div_nonzero. The second division will be only used to facilitate our
    // transformations.
    if (divisor->isConstant() && multiplier->isConstant()) {
      // If both are scalar we may be able to find a common factor.
      if (immediateEquals(evaluateOp(new Mod(multiplier, divisor)), 0)) {
        // The common factor becomes 'scalar' of the term, e.g.,in t/3%7*6,
        // divisor=multiplier=3, scalar=2.
        Expr* c = evaluateOp(new Div(multiplier, divisor));
        scalar = c;
      } else if (immediateEquals(evaluateOp(new Mod(divisor, multiplier)), 0)) {
        // The common factor becomes part of 'denom', e.g., in t/14%7*2,
        // divisor=multiplier=2, denom=t/7.
        Expr* c = evaluateOp(new Div(divisor, multiplier));
        divisor = multiplier;
        denom = IRSimplifier::simplify(new Div(other, c));
      } else {
        return c10::nullopt;
      }
    } else {
      return c10::nullopt;
    }
  }

  // Deny cases in which divisor=1. Such cases are considered as Mods.
  if (divisor->isConstant() && immediateEquals(divisor, 1)) {
    return c10::nullopt;
  }

  if (!scalar) {
    scalar = getImmediateByType(multiplier->dtype(), 1);
  }

  return new ModRound(scalar, denom, divisor, mod_divisor);
}

// Search the polynomial for Terms that can be merged in
// (1) Round + Mod pattern: (x/y) * y + x % y => RoundOff(x,y) + Mod(x, y) => x
// (2) Mod round + Mod pattern: (x/y % z)*y + x%y => ModRound(x, y, z) + Mod(x,
// y) => x % (y*z)
const Expr* simplifyRoundModPattern(const Polynomial* poly) {
  std::vector<const Term*> rounds;
  std::vector<const Term*> mods;
  std::vector<const Term*> mod_rounds;
  std::vector<const Term*> others;

  // Split out the Mod, ModRounds and RoundOffs operations so we can inspect.
  for (auto* c : poly->variables()) {
    if (c->variables().size() > 1) {
      if (auto a = isModRound(c)) {
        mod_rounds.push_back(c);
      } else {
        others.push_back(c);
      }
      continue;
    }

    const Expr* e = c->variables()[0];

    if (dynamic_cast<const RoundOff*>(e)) {
      rounds.push_back(c);
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    } else if (e->expr_type() == IRNodeType::kMod) {
      if (auto a = isModRound(c)) {
        mod_rounds.push_back(c);
      } else {
        mods.push_back(c);
      }
    } else {
      others.push_back(c);
    }
  }

  // Can't continue without at least one RoundOff/ModRound and one Mod.
  if ((rounds.empty() && mod_rounds.empty()) || mods.empty()) {
    return nullptr;
  }

  HashProvider& hasher = poly->hasher();
  bool didAnything = false;
  std::vector<const Term*> mods_merged;
  bool repeat = true;
  // Repeat merging terms till there are no Mods or the terms cannot be merged
  // any further.
  while (!mods.empty() && repeat) {
    repeat = false;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (int i = mods.size() - 1; i >= 0; i--) {
      const Term* m = mods[i];
      const Mod* mod = dynamic_cast<const Mod*>(m->variables()[0]);
      CHECK(mod);
      const Expr* mod_lhs = IRSimplifier::simplify(mod->lhs());
      const Expr* mod_rhs = IRSimplifier::simplify(mod->rhs());
      bool merged = false;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      for (int j = mod_rounds.size() - 1; j >= 0; j--) {
        const Term* mr = mod_rounds[j];
        auto a = isModRound(mr);
        CHECK(a);
        const ModRound* mod_round = dynamic_cast<const ModRound*>(*a);

        // TODO: for now don't attempt partial factorization of this
        // optimization. E.g. it's possible to do: 2 * (x/y%z) * y + (x%y) =>
        // x%(y*z) + (x/y%z) * y
        if (!immediateEquals(
                evaluateOp(new Sub(mod_round->scalar, m->scalar())), 0)) {
          continue;
        }
        // Valid optimization if mod LHS matches denom and mod RHS matches
        // divisor.
        if (hasher.hash(mod_round->denom) == hasher.hash(mod_lhs) &&
            hasher.hash(mod_round->divisor) == hasher.hash(mod_rhs)) {
          // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
          const Term* merged_m = new Term(
              hasher,
              mod_round->scalar,
              IRSimplifier::simplify(new Mod(
                  mod_round->denom,
                  new Mul(mod_round->divisor, mod_round->mod_divisor))));
          mods_merged.push_back(merged_m);
          merged = true;
          repeat = true;
          didAnything = true;
          mods.erase(mods.begin() + i);
          mod_rounds.erase(mod_rounds.begin() + j);
          break;
        }
      }

      if (merged) {
        continue;
      }

      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      for (int k = rounds.size() - 1; k >= 0; k--) {
        const Term* r = rounds[k];
        const RoundOff* roundoff =
            dynamic_cast<const RoundOff*>(r->variables()[0]);
        CHECK(roundoff);

        // TODO: for now don't attempt partial factorization of this
        // optimization. E.g. it's possible to do: 2 * (x/y) * y + (x%y) => x +
        // (x/y) * y but unsure thats actually much better, particulary with
        // CSE.
        if (!immediateEquals(
                evaluateOp(new Sub(r->scalar(), m->scalar())), 0)) {
          continue;
        }
        const Expr* round_lhs = IRSimplifier::simplify(roundoff->lhs());
        const Expr* round_rhs = IRSimplifier::simplify(roundoff->rhs());
        // Valid optimization if LHS and RHS are equal for both.
        if (hasher.hash(round_lhs) == hasher.hash(mod_lhs) &&
            hasher.hash(round_rhs) == hasher.hash(mod_rhs)) {
          const Term* merged_r = new Term(hasher, r->scalar(), round_lhs);
          others.push_back(merged_r);
          merged = true;
          didAnything = true;
          mods.erase(mods.begin() + i);
          rounds.erase(rounds.begin() + k);
          break;
        }
      }

      // If we didn't merge, move out the Mod.
      if (!merged) {
        others.push_back(m);
        mods.erase(mods.begin() + i);
      }

    } // end of for-loop

    // Add newly generated Mods for merging opportunities in the next iteration.
    if (!mods_merged.empty()) {
      mods.insert(mods.end(), mods_merged.begin(), mods_merged.end());
      mods_merged.clear();
    }

  } // end of while-loop

  // If we made no changes, just exit.
  if (!didAnything) {
    return nullptr;
  }

  // Keep remaining ModRounds and RoundOffs.
  if (!mod_rounds.empty()) {
    others.insert(others.end(), mod_rounds.begin(), mod_rounds.end());
  }

  if (!rounds.empty()) {
    others.insert(others.end(), rounds.begin(), rounds.end());
  }

  return new Polynomial(hasher, poly->scalar(), others);
}

// Trivially factorize terms by GCD of scalar components.
const Term* IRSimplifierBase::factorizePolynomial(const Polynomial* poly) {
  const Expr* scalar = poly->scalar();
  const std::vector<const Term*>& variables = poly->variables();

  // Compute the GCD of terms.
  const Expr* GCD = polyGCD(poly);

  // No GCD means 0 or 1 and can't be factored.
  if (!GCD) {
    return nullptr;
  }

  // Create new struture.
  std::vector<const Term*> newPolyTerms;
  for (auto* t : variables) {
    // New term with the scalar divided by the GCD.
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    newPolyTerms.push_back(new Term(
        poly->hasher(), evaluateOp(new Div(t->scalar(), GCD)), t->variables()));
  }

  Polynomial* newPoly = new Polynomial(
      poly->hasher(), evaluateOp(new Div(scalar, GCD)), newPolyTerms);

  return new Term(poly->hasher(), GCD, newPoly);
}

const Expr* TermExpander::mutate(const Polynomial* v) {
  if (v->variables().empty()) {
    return v->scalar();
  }

  // If this Polynomial can be factorized: do it, then expand the result.
  if (const Expr* simplified = simplifyRoundModPattern(v)) {
    return simplified->accept_mutator(this);
  }

  // If this Polynomial can be factorized: do it, then expand the result.
  if (const Expr* factorized = factorizePolynomial(v)) {
    return factorized->accept_mutator(this);
  }

  std::vector<const Term*> addTerms;
  std::vector<const Term*> subTerms;

  // partition the terms into a list to add and list to subtract.
  for (auto* node : v->variables()) {
    if (immediateIsNegative(node->scalar())) {
      subTerms.push_back(node);
    } else if (!immediateEquals(node->scalar(), 0)) {
      addTerms.push_back(node);
    }
    // Skip terms with a scalar of zero.
  }

  // The last node constructed.
  const Expr* lastNode{nullptr};

  for (auto* node : addTerms) {
    const Expr* simpleNode = node->accept_mutator(this);

    if (lastNode == nullptr) {
      lastNode = simpleNode;
      continue;
    }

    if (isMultilanePrimitive(simpleNode)) {
      auto* ret = combineMultilane<Add>(lastNode, simpleNode);
      if (ret) {
        // simplify result first, then expand.
        lastNode = ret->accept_mutator(simplifier_);
        lastNode = lastNode->accept_mutator(this);
        continue;
      }
    }

    lastNode = new Add(lastNode, simpleNode);
  }

  // If we have no add terms the scalar should go first.
  // E.g. 1 - x.
  bool scalarWritten = false;
  if (lastNode == nullptr) {
    auto* scalarNode = v->scalar()->accept_mutator(simplifier_);

    if (!immediateEquals(scalarNode, 0)) {
      lastNode = scalarNode;
      scalarWritten = true;
    }
  }

  for (auto* node : subTerms) {
    // Can still be first node if scalarVal is 0.
    if (lastNode == nullptr) {
      lastNode = node->accept_mutator(this);
      continue;
    }

    // Negate the term back to positive since we'll be subtracting it.
    const Expr* negated = evaluateOp(new Mul(
        getImmediateByType(node->scalar()->dtype(), -1), node->scalar()));
    Term* newRHS = new Term(node->hasher(), negated, node->variables());
    lastNode = new Sub(lastNode, newRHS->accept_mutator(this));
  }

  if (scalarWritten || immediateEquals(v->scalar(), 0)) {
    if (!lastNode) {
      return getImmediateByType(v->dtype(), 0);
    }
    return lastNode;
  }

  if (immediateIsNegative(v->scalar())) {
    // Negate the scalar and subtract.
    const Expr* negated = evaluateOp(
        new Mul(getImmediateByType(lastNode->dtype(), -1), v->scalar()));
    lastNode = new Sub(lastNode, evaluateOp(negated));
  } else {
    // we want to avoid a cast to the scalar if it would happen.
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (v->scalar()->dtype() != lastNode->dtype()) {
      lastNode = new Add(
          lastNode, evaluateOp(new Cast(lastNode->dtype(), v->scalar())));
    } else {
      lastNode = new Add(lastNode, v->scalar());
    }
  }

  return lastNode;
}

const Expr* TermExpander::mutate(const MaxTerm* v) {
  const auto& variables = v->variables();
  if (variables.empty()) {
    if (!v->scalar()) {
      // This case should never happen because MaxTerm will be created only
      // on valid Max expressions.
      throw std::logic_error("empty maxterm op");
    }
    return v->scalar();
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const Expr* max;
  if (v->scalar()) {
    max = new Max(variables[0], v->scalar(), v->propagate_nans());
  } else {
    max = variables[0];
  }
  for (size_t i = 1; i < variables.size(); i++) {
    max = new Max(max, variables[i], v->propagate_nans());
  }
  return max->accept_mutator(this);
}

const Expr* TermExpander::mutate(const MinTerm* v) {
  const auto& variables = v->variables();
  if (variables.empty()) {
    if (!v->scalar()) {
      // This case should never happen because MinTerm will be created only
      // on valid Min expressions.
      throw std::logic_error("empty minterm op");
    }
    return v->scalar();
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const Expr* min;
  if (v->scalar()) {
    min = new Min(variables[0], v->scalar(), v->propagate_nans());
  } else {
    min = variables[0];
  }
  for (size_t i = 1; i < variables.size(); i++) {
    min = new Min(min, variables[i], v->propagate_nans());
  }
  return min->accept_mutator(this);
}

// Expands RoundOff(x, y) => Term(1, Div(x, y), y), which will later be expanded
// to Mul(Div(x, y), y).
const Expr* TermExpander::mutate(const RoundOff* v) {
  Term* term = new Term(
      simplifier_->hasher(),
      getImmediateByType(v->dtype(), 1),
      new Div(v->lhs(), v->rhs()),
      v->rhs());
  return term->accept_mutator(this);
}

const Expr* buf_flat_size(const Buf* v) {
  std::vector<const Expr*> dims = v->dims();

  const Expr* flattened = getImmediateByType(kInt, 1);
  for (auto& dim : dims) {
    flattened = new Mul(flattened, dim);
  }
  flattened = IRSimplifier::simplify(flattened);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return flattened;
}

Stmt* TermExpander::mutate(const Allocate* v) {
  const Buf* buf = v->buf();
  const Buf* buf_new = dynamic_cast<const Buf*>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  const Expr* flattened = buf_flat_size(buf_new);

  if (flattened->isConstant() && immediateEquals(flattened, 0)) {
    eliminated_allocations_.insert(buf_new->base_handle());
    return nullptr;
  }

  if (buf_new == buf) {
    return (Stmt*)v;
  }

  return new Allocate(buf_new);
}

Stmt* TermExpander::mutate(const Free* v) {
  const Buf* buf = v->buf();
  const Buf* buf_new = dynamic_cast<const Buf*>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);

  if (eliminated_allocations_.count(buf_new->base_handle())) {
    eliminated_allocations_.erase(buf_new->base_handle());
    return nullptr;
  }

  if (buf_new == buf) {
    return (Stmt*)v;
  }

  return new Free(buf_new);
}

// Combines adjactent Cond nodes with identical conditions.
Block* TermExpander::fuseConditions(Block* v) {
  std::vector<Stmt*> stmts;
  bool did_anything = false;
  Cond* prev_cond = nullptr;

  for (auto* s : *v) {
    Cond* cond = dynamic_cast<Cond*>(s);
    if (!cond) {
      prev_cond = nullptr;
      stmts.push_back(s);
      continue;
    }

    // If the previous statement is a Cond and the conditions are identical,
    // then we fuse.
    if (!prev_cond ||
        hasher_.hash(prev_cond->condition()) !=
            hasher_.hash(cond->condition())) {
      prev_cond = cond;
      stmts.push_back(s);
      continue;
    }

    // Fuse the two Conds by appending the bodies of the second Cond to the
    // first.
    Block* true_block = new Block({});
    Block* false_block = new Block({});

    if (prev_cond->true_stmt()) {
      true_block->splice(true_block->end(), prev_cond->true_stmt());
    }

    if (cond->true_stmt()) {
      true_block->splice(true_block->end(), cond->true_stmt());
    }

    if (prev_cond->false_stmt()) {
      false_block->splice(false_block->end(), prev_cond->false_stmt());
    }

    if (cond->false_stmt()) {
      false_block->splice(false_block->end(), cond->false_stmt());
    }

    // avoid unflattening this Cond if we can.
    if (true_block->empty()) {
      true_block = nullptr;
    }

    if (false_block->empty()) {
      false_block = nullptr;
    }

    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    Stmt* new_cond = prev_cond->cloneWithNewBodies(true_block, false_block)
                         ->accept_mutator(this);
    prev_cond = dynamic_cast<Cond*>(new_cond);

    // erase, which shortens the list.
    stmts.pop_back();
    stmts.push_back(new_cond);
    did_anything = true;
  }

  if (!did_anything) {
    return v;
  }

  // clean up parents.
  for (auto* s : stmts) {
    if (s->get_parent() == v) {
      v->remove_stmt(s);
    }
  }

  return new Block(stmts);
}

Stmt* TermExpander::fuseSyncThreads(Block* block) {
  // only really first if highest level Block.
  bool first = block->get_parent() == nullptr;
  SyncThreads* last = nullptr;
  std::vector<Stmt*> stmts;
  bool did_anything = false;

  for (auto* s : *block) {
    SyncThreads* sync = dynamic_cast<SyncThreads*>(s);
    if (!sync) {
      first = false;
      last = nullptr;
      stmts.push_back(s);
      continue;
    }

    if (first || last) {
      did_anything = true;
      continue;
    }

    last = sync;
    first = false;
    stmts.push_back(s);
  }

  if (last) {
    stmts.pop_back();
    did_anything = true;
  }

  if (!did_anything) {
    return block;
  }

  // clean up parents.
  for (auto* s : stmts) {
    if (s->get_parent() == block) {
      block->remove_stmt(s);
    }
  }

  return new Block({stmts});
}

Stmt* TermExpander::mutate(const Block* v) {
  Stmt* new_stmt = IRSimplifierBase::mutate(v);
  Block* new_block = dynamic_cast<Block*>(new_stmt);
  if (!new_block) {
    return new_stmt;
  }

  // fuseConditions will return the original block if it cannot fuse.
  new_block = fuseConditions(new_block);
  /// fuseSyncThreads too.
  return fuseSyncThreads(new_block);
}

bool exprEquals(const Expr* A, const Expr* B) {
  try {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    const Expr* diff = IRSimplifier::simplify(new Sub(A, B));
    if (!diff->isConstant()) {
      return false;
    }
    return immediateEquals(diff, 0);
  } catch (std::exception& e) {
    return false;
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
