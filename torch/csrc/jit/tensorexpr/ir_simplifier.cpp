#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
          new Op(bc->value(), ramp->base()), ramp->stride(), ramp->lanes());
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
    return variable;
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
      return ret->accept_mutator(this);
    }
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
    const Expr* negateScalar = evaluateOp(new Mul(minusOne, lhsTerm->scalar()));

    std::vector<const Term*> variables;
    for (auto* t : lhsPoly->variables()) {
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
      return ret->accept_mutator(this);
    }
  }

  // If this is a floating point Mul then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return new Mul(lhs_new, rhs_new);
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

  const Expr* scalar = nullptr;
  const Expr* variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = lhs_new;
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = rhs_new;
    variable = lhs_new;
  }

  if (scalar && lhsTerm) {
    const Expr* newScalar = evaluateOp(new Mul(scalar, lhsTerm->scalar()));
    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }
    return new Term(hasher_, newScalar, lhsTerm->variables());
  }

  if (scalar && rhsTerm) {
    const Expr* newScalar = evaluateOp(new Mul(scalar, rhsTerm->scalar()));

    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }
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
    std::vector<const Expr*> vars = lhsTerm->variables();
    vars.push_back(rhs_new);
    return new Term(hasher_, lhsTerm->scalar(), vars);
  }
  if (rhsTerm) {
    std::vector<const Expr*> vars = rhsTerm->variables();
    vars.push_back(lhs_new);
    return new Term(hasher_, rhsTerm->scalar(), vars);
  }

  // Two variables, create a new Term.
  return new Term(hasher_, getImmediateByType(v->dtype(), 1), lhs_new, rhs_new);
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

  return new Cast(v->dtype(), node);
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
      if (v->scalar()->dtype() != lastNode->dtype()) {
        lastNode = new Mul(
            evaluateOp(new Cast(lastNode->dtype(), v->scalar())), lastNode);
      } else {
        lastNode = new Mul(v->scalar(), lastNode);
      }
    } else {
      lastNode = v->scalar();
    }
  }

  return lastNode;
}

// Simple recursive GCD.
template <typename T>
T gcd(T a, T b) {
  if (b == 0) {
    return a;
  }
  return gcd(b, a % b);
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
  long GCD = immediateAs<long>(scalar);
  for (auto* t : variables) {
    long termScalar = immediateAs<long>(t->scalar());
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

// Trivially factorize terms by GCD of scalar components.
const Expr* TermExpander::factorizePolynomial(const Polynomial* poly) {
  const Expr* scalar = poly->scalar();
  const std::vector<const Term*>& variables = poly->variables();
  bool floatScalars = false;

  // Check types.
  for (auto& p : variables) {
    if (is_floating_point(p->dtype().scalar_type()) ||
        is_floating_point(p->scalar()->dtype().scalar_type())) {
      floatScalars = true;
    }
  }
  if (is_floating_point(scalar->dtype().scalar_type())) {
    floatScalars = true;
  }

  // floating point isn't generally distributive.
  if (floatScalars) {
    return nullptr;
  }

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
    if (v->scalar()->dtype() != lastNode->dtype()) {
      lastNode = new Add(
          lastNode, evaluateOp(new Cast(lastNode->dtype(), v->scalar())));
    } else {
      lastNode = new Add(lastNode, v->scalar());
    }
  }

  return lastNode;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
