#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

#include <utility>

namespace torch::jit::tensorexpr {

// Creates a new Expr of the given type with the provided lhs and rhs.
inline ExprPtr newBinaryOpOfType(
    IRNodeType expr_type,
    ExprPtr lhs,
    ExprPtr rhs,
    bool option) {
  switch (expr_type) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case IRNodeType::kAdd:
      return alloc<Add>(lhs, rhs);
    case IRNodeType::kSub:
      return alloc<Sub>(lhs, rhs);
    case IRNodeType::kMul:
      return alloc<Mul>(lhs, rhs);
    case IRNodeType::kDiv:
      return alloc<Div>(lhs, rhs);
    case IRNodeType::kMod:
      return alloc<Mod>(lhs, rhs);
    case IRNodeType::kMax:
      return alloc<Max>(lhs, rhs, option);
    case IRNodeType::kMin:
      return alloc<Min>(lhs, rhs, option);
    case IRNodeType::kAnd:
      return alloc<And>(lhs, rhs);
    case IRNodeType::kXor:
      return alloc<Xor>(lhs, rhs);
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs, rhs);
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs, rhs);
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
      return nullptr;
  }
}

template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutateBinaryOp(
    NodePtr<Op> v,
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  ExprPtr rhs_new = rhs->accept_mutator(mutator);

  ExprPtr node = v;

  if (lhs != lhs_new || rhs != rhs_new) {
    node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
  }

  // Can only fold if both sides are constant.
  if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
    return node;
  }

  return evaluateOp(node);
}

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
static bool isMultilanePrimitive(ExprPtr e) {
  return to<Broadcast>(e) || to<Ramp>(e);
}

SimplifierHashType Term::hashVars() const {
  SimplifierHashType hash;
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }

  return hash;
}

void Term::sort() {
  // order of ops important for float
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

SimplifierHashType Polynomial::hashVars() const {
  SimplifierHashType hash;
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }
  return hash;
}

void Polynomial::sort() {
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

void MaxTerm::uniquefy() {
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    return hasher_.hash(a) < hasher_.hash(b);
  });
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // Once we removed duplicates, sort terms alphabetically for stability.
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

void MinTerm::uniquefy() {
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    return hasher_.hash(a) < hasher_.hash(b);
  });
  auto it = std::unique(
      variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // Once we removed duplicates, sort terms alphabetically for stability.
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(variables_.begin(), variables_.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });
}

// Handles optimization cases for Broadcast/Ramp +/- Broadcast/Ramp
template <class Op>
ExprPtr combineMultilane(ExprPtr lhs, ExprPtr rhs) {
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Broadcast>(
          alloc<Op>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (RampPtr r = to<Ramp>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(bc->value(), r->base()), r->stride(), r->lanes());
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {
    if (RampPtr rother = to<Ramp>(rhs)) {
      if (ramp->lanes() != rother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), rother->base()),
          alloc<Op>(ramp->stride(), rother->stride()),
          ramp->lanes());
      return ret;
    }

    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }
      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), bc->value()), ramp->stride(), ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

// Handles optimization cases for Broadcast/Ramp * Broadcast/Ramp
static ExprPtr mulMultilane(ExprPtr lhs, ExprPtr rhs) {
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Broadcast>(
          alloc<Mul>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (RampPtr r = to<Ramp>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), r->base()),
          alloc<Mul>(bc->value(), r->stride()),
          r->lanes());
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {
    if (RampPtr r = to<Ramp>(rhs)) {
      if (ramp->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(ramp->base(), r->base()),
          alloc<Mul>(ramp->stride(), r->stride()),
          r->lanes());
      return ret;
    }

    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), ramp->base()),
          alloc<Mul>(bc->value(), ramp->stride()),
          ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

void PolynomialTransformer::addOrUpdateTerm(
    std::unordered_map<SimplifierHashType, TermPtr>& varmap,
    TermPtr term) {
  SimplifierHashType hash = term->hashVars();
  auto insertRes = varmap.emplace(hash, term);
  if (insertRes.second == false) {
    TermPtr lt = insertRes.first->second;
    ExprPtr termScalar = evaluateOp(alloc<Add>(lt->scalar(), term->scalar()));

    // If the term is canceled out, remove from the map.
    if (immediateEquals(termScalar, 0)) {
      varmap.erase(hash);
      return;
    }

    varmap[hash] = alloc<Term>(hasher_, termScalar, lt->variables());
  }
}

ExprPtr PolynomialTransformer::addPolynomials(
    PolynomialPtr lhs,
    PolynomialPtr rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  for (const auto& lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }
  for (const auto& rt : rhs->variables()) {
    addOrUpdateTerm(varmap, rt);
  }

  ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

// Insert a new Term into the provided polynomial. If the new term has common
// variables to an existing term it is combined.
ExprPtr PolynomialTransformer::insertTerm(PolynomialPtr poly, TermPtr term) {
  SimplifierHashType tHash = term->hashVars();
  std::vector<TermPtr> newVars;

  bool found = false;
  for (const auto& v : poly->variables()) {
    if (v->hashVars() == tHash) {
      ExprPtr newScalar = evaluateOp(alloc<Add>(term->scalar(), v->scalar()));
      found = true;
      // Skip this term if we cancelled it out.
      if (immediateEquals(newScalar, 0)) {
        continue;
      }
      auto term = alloc<Term>(hasher_, newScalar, v->variables());
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

  auto Poly = alloc<Polynomial>(hasher_, poly->scalar(), newVars);
  return Poly;
}

ExprPtr PolynomialTransformer::mutate(AddPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    ExprPtr result = evaluateOp(alloc<Add>(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = combineMultilane<Add>(lhs_new, rhs_new)) {
      return ret->accept_mutator(this);
    }
  }

  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
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
    auto c = alloc<Cast>(v->dtype(), variable);
    return c->accept_mutator(this);
  }

  // If this is a floating point Add then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Add>(lhs_new, rhs_new);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    return addPolynomials(lhsPoly, rhsPoly);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return insertTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return insertTerm(rhsPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    // If the terms refer to the same variables: combine them.
    if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
      ExprPtr newScalar =
          evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar()));

      // If the terms cancelled out, return zero.
      if (immediateEquals(newScalar, 0)) {
        return newScalar->accept_mutator(this);
      }

      return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
    }

    // Otherwise this is a new polynomial with no scalar and two variable
    // terms.
    return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
  }

  // Adds are commutative.
  PolynomialPtr poly = lhsPoly ? lhsPoly : rhsPoly;

  // Add to Polynomial->scalar().
  if (scalar && poly) {
    ExprPtr newScalar = evaluateOp(alloc<Add>(scalar, poly->scalar()));
    return alloc<Polynomial>(hasher_, newScalar, poly->variables());
  }

  // Simple Polynomial with a scalar and Term.
  TermPtr term = lhsTerm ? lhsTerm : rhsTerm;
  if (scalar && term) {
    return alloc<Polynomial>(hasher_, scalar, term);
  }

  // Simple Term with a scalar and variable type.
  if (scalar) {
    return alloc<Polynomial>(
        hasher_, scalar, alloc<Term>(hasher_, immLike(v, 1), variable));
  }

  // If LHS is neither Term not Polynomial, wrap it in a Term.
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
  }

  // Same for RHS.
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, 1), rhs_new);
  }

  // If we now have a poly and a term, we can insert.
  if (poly) {
    return insertTerm(poly, lhsTerm ? lhsTerm : rhsTerm);
  }

  if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
    return alloc<Term>(
        hasher_,
        evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar())),
        lhsTerm->variables());
  }

  // If all else fails we have a new Polynomial with two new variable Terms.
  return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
}

ExprPtr PolynomialTransformer::subTerms(
    TermPtr lhs,
    TermPtr rhs,
    bool negated) {
  // If RHS not already negated, negate it.
  if (!negated) {
    ExprPtr minusOne = immLike(rhs, -1);
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhs->scalar()));
    rhs = alloc<Term>(hasher_, negateScalar, rhs->variables());
  }

  if (lhs->hashVars() == rhs->hashVars()) {
    ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));

    // If the terms cancel out, return zero.
    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }

    return alloc<Term>(hasher_, newScalar, lhs->variables());
  }

  return alloc<Polynomial>(
      hasher_,
      getImmediateByType(promoteTypes(lhs->dtype(), rhs->dtype()), 0),
      lhs,
      rhs);
}

// Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
// possible.
ExprPtr PolynomialTransformer::subPolynomials(
    PolynomialPtr lhs,
    PolynomialPtr rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  for (const auto& lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }

  for (const auto& rt : rhs->variables()) {
    // Polynomials add their terms, so negate the RHS's Terms.
    ExprPtr negated = evaluateOp(alloc<Mul>(immLike(rt, -1), rt->scalar()));
    TermPtr newRHS = alloc<Term>(hasher_, negated, rt->variables());
    addOrUpdateTerm(varmap, newRHS);
  }

  ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs->scalar(), rhs->scalar()));

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
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

ExprPtr PolynomialTransformer::mutate(SubPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    ExprPtr result = evaluateOp(alloc<Sub>(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = combineMultilane<Sub>(lhs_new, rhs_new)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);
    }
  }

  if (rhs_new->isConstant() && immediateEquals(rhs_new, 0)) {
    auto c = alloc<Cast>(v->dtype(), lhs_new);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);
  }

  // If this is a floating point Sub then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Sub>(lhs_new, rhs_new);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    auto ret = subPolynomials(lhsPoly, rhsPoly);
    if (!ret) {
      // Cancelled out completely.
      return immLike(v, 0);
    }
    return ret;
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  // Polynomial - Term.
  if (lhsPoly && rhsTerm) {
    // Negate the term.
    ExprPtr negate =
        evaluateOp(alloc<Mul>(immLike(rhsTerm, -1), rhsTerm->scalar()));
    TermPtr newTerm = alloc<Term>(hasher_, negate, rhsTerm->variables());
    return insertTerm(lhsPoly, newTerm);
  }

  // Term - Polynomial.
  if (rhsPoly && lhsTerm) {
    // Negate every part of the Polynomial.
    ExprPtr minusOne = immLike(lhsTerm, -1);
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    PolynomialPtr newPoly = alloc<Polynomial>(hasher_, negateScalar, variables);
    return insertTerm(newPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    return subTerms(lhsTerm, rhsTerm, false);
  }

  bool lhsScalar = lhs_new->isConstant();
  bool rhsScalar = rhs_new->isConstant();

  if (lhsPoly && rhsScalar) {
    // Easy path, just sub the scalar component.
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhsPoly->scalar(), rhs_new));
    return alloc<Polynomial>(hasher_, newScalar, lhsPoly->variables());
  }

  if (lhsScalar && rhsPoly) {
    // Sub the scalar component.
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs_new, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    ExprPtr minusOne = immLike(rhsPoly, -1);
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    return alloc<Polynomial>(hasher_, newScalar, variables);
  }

  if (lhsTerm && rhsScalar) {
    // Negate the constant.
    ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
    return alloc<Polynomial>(hasher_, negate, lhsTerm);
  }

  if (lhsScalar && rhsTerm) {
    // Negate the RHS Term.
    ExprPtr negate = evaluateOp(
        alloc<Mul>(immLike(rhsTerm->scalar(), -1), rhsTerm->scalar()));

    return alloc<Polynomial>(
        hasher_, lhs_new, alloc<Term>(hasher_, negate, rhsTerm->variables()));
  }

  // simple term with a scalar and variable type.
  if (lhsScalar) {
    // Create a negated term.
    return alloc<Polynomial>(
        hasher_, lhs_new, alloc<Term>(hasher_, immLike(v, -1), rhs_new));
  }

  if (rhsScalar) {
    // Negate the scalar.
    ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
    return alloc<Polynomial>(
        hasher_, negate, alloc<Term>(hasher_, immLike(v, 1), lhs_new));
  }

  // no scalar...
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
  }

  bool createdRHSnegated = false;
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, -1), rhs_new);
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
    ExprPtr minusOne = immLike(rhsPoly, -1);
    ExprPtr newScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    auto poly = alloc<Polynomial>(hasher_, newScalar, variables);
    return insertTerm(poly, lhsTerm);
  }

  return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
}

// Multiply two terms together, usually creating a new term with the variable
// lists concatenated.
TermPtr PolynomialTransformer::mulTerms(TermPtr lhs, TermPtr rhs) {
  ExprPtr scalar = evaluateOp(alloc<Mul>(lhs->scalar(), rhs->scalar()));
  if (immediateEquals(scalar, 0)) {
    return nullptr;
  }

  // Can reorder here since floating point ops don't get put into Terms.
  std::vector<ExprPtr> variables;
  std::vector<ExprPtr> multilaneVariables;
  // For now don't handle exponents.
  for (const auto& c : lhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }
  for (const auto& c : rhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }

  // Merge all the multilane vars:
  ExprPtr lastNode{nullptr};
  for (const auto& node : multilaneVariables) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      if (auto next = mulMultilane(lastNode, node)) {
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

  return alloc<Term>(hasher_, scalar, variables);
}

// Multiply a Polynomial by a Term.
ExprPtr PolynomialTransformer::polyByTerm(PolynomialPtr poly, TermPtr term) {
  // poly * term
  //    = (poly_terms + poly_scalar) * term
  //    = poly_terms * term + poly_scalar * term

  // First, multiply all variables (terms) in the polynomial by the input
  // term.
  std::vector<TermPtr> newTerms;
  for (const auto& var : poly->variables()) {
    TermPtr newTerm = mulTerms(var, term);
    if (newTerm) {
      newTerms.push_back(newTerm);
    }
  }

  // If the scalar in poly is not 0, it must be multiplied by term.
  // If there are no variables in term, this becomes the scalar in the result
  // polynomial. If there are variables in term, this becomes a new term in
  // the result polynomial.
  if (!immediateEquals(poly->scalar(), 0)) {
    ExprPtr scalar = evaluateOp(alloc<Mul>(poly->scalar(), term->scalar()));
    if (term->variables().empty()) {
      return alloc<Polynomial>(hasher_, scalar, newTerms);
    }
    newTerms.push_back(alloc<Term>(hasher_, scalar, term->variables()));
  }

  // The only case when the result polynomial has a scalar is when the input
  // term does not have any variables and the input polynomial has a non-zero
  // scalar. That case is handled above. So, at this point, we do not have any
  // scalars in the result polynomial.
  return alloc<Polynomial>(hasher_, std::move(newTerms));
}

// Does multiplying these two expressions make a Rounding Off operation.
// e.g. LHS = (x/y),  RHS = y => (x / y) * y => RoundOff(x, y).
ExprPtr PolynomialTransformer::isRoundOff(ExprPtr lhs, ExprPtr rhs) {
  DivPtr div{nullptr};
  ExprPtr other{nullptr};

  if ((div = to<Div>(lhs))) {
    other = rhs;
  } else if ((div = to<Div>(rhs))) {
    other = lhs;
  } else {
    return nullptr;
  }

  ExprPtr denom = div->rhs();

  if (TermPtr denomTerm = to<Term>(denom)) {
    if (immediateEquals(denomTerm->scalar(), 1) &&
        denomTerm->variables().size() == 1) {
      denom = denomTerm->variables()[0];
    }
  }

  if (hasher_.hash(denom) == hasher_.hash(other)) {
    // If the denominator is equal to the other, then yes it's a RoundOff.
    return alloc<RoundOff>(div->lhs(), div->rhs());
  }

  if (denom->isConstant() && other->isConstant()) {
    if (immediateEquals(denom, 0) || immediateEquals(other, 0)) {
      return nullptr;
    }
    // If they are both scalar we may be able to find a common factor.
    if (immediateEquals(evaluateOp(alloc<Mod>(other, denom)), 0)) {
      ExprPtr scalar = evaluateOp(alloc<Div>(other, denom));
      ExprPtr newDenom = evaluateOp(alloc<Div>(other, scalar));
      return alloc<Term>(
          hasher_, scalar, alloc<RoundOff>(div->lhs(), newDenom));
    }
  }

  return nullptr;
}

// Inserts a new component into a term, looking for opportunities to simplify.
ExprPtr PolynomialTransformer::insertIntoTerm(TermPtr term, ExprPtr expr) {
  std::vector<ExprPtr> vars;

  // Search for RoundOffs.
  bool merged{false};
  for (const auto& component : term->variables()) {
    if (auto roundoff = isRoundOff(component, expr)) {
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

  return alloc<Term>(hasher_, term->scalar(), vars);
}

ExprPtr PolynomialTransformer::mutate(MulPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Mul>(lhs_new, rhs_new));
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = mulMultilane(lhs_new, rhs_new)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return ret->accept_mutator(this);
    }
  }

  // Order doesn't matter.
  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
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
    auto c = alloc<Cast>(v->dtype(), variable);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c->accept_mutator(this);
  }

  // If this is a floating point Mul then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Mul>(lhs_new, rhs_new);
  }

  // Handle special case mul by 0.
  if (scalar && immediateEquals(scalar, 0)) {
    return immLike(v, 0);
  }

  // Catch cases of rounding (Div(A/B) * B).
  if (auto ret = isRoundOff(lhs_new, rhs_new)) {
    return ret;
  } else if (auto ret = isRoundOff(v->lhs(), v->rhs())) {
    // We can break the Round + Mod pattern via factorization of the Div, so
    // check whether it would have worked on the unsimplified tree. If so, we
    // need to simplify again.
    return ret->accept_mutator(this);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    // This expands to more terms that we can't generally fix without variable
    // factorization, it's more efficient to just leave these as Muls.
    return alloc<Mul>(lhsPoly, rhsPoly);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

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
    ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, lhsTerm->scalar()));
    return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
  }

  if (scalar && rhsTerm) {
    ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, rhsTerm->scalar()));
    return alloc<Term>(hasher_, newScalar, rhsTerm->variables());
  }

  // If this is a scalar * a Polynomial, push the scalar term down.
  // We can wrap the scalar with a Term and use polyByTerm.
  if (scalar && lhsPoly) {
    return polyByTerm(lhsPoly, alloc<Term>(hasher_, scalar));
  }
  if (scalar && rhsPoly) {
    return polyByTerm(rhsPoly, alloc<Term>(hasher_, scalar));
  }

  // simple term with a scalar and variable type.
  if (scalar) {
    return alloc<Term>(hasher_, scalar, variable);
  }

  // Multiplying Polynomial by variable can be wrapped in a term and handled
  // by polyByTerm also.
  if (lhsPoly) {
    auto term = alloc<Term>(hasher_, immLike(rhs_new, 1), rhs_new);
    return polyByTerm(lhsPoly, term);
  }
  if (rhsPoly) {
    auto term = alloc<Term>(hasher_, immLike(lhs_new, 1), lhs_new);
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
  return alloc<Term>(hasher_, immLike(v, 1), lhs_new, rhs_new);
}

static ExprPtr factorizeDivision(ExprPtr lhs_new, ExprPtr rhs_new) {
  if (!lhs_new || !rhs_new) {
    return nullptr;
  }

  ExprPtr leftScalar = lhs_new->isConstant() ? lhs_new : nullptr;
  ExprPtr rightScalar = rhs_new->isConstant() ? rhs_new : nullptr;

  auto lhsTerm = to<Term>(lhs_new);
  auto rhsTerm = to<Term>(rhs_new);
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

  leftScalar = evaluateOp(alloc<Div>(leftScalar, immLike(leftScalar, GCD)));
  rightScalar = evaluateOp(alloc<Div>(rightScalar, immLike(rightScalar, GCD)));

  if (lhsTerm) {
    lhs_new = alloc<Term>(lhsTerm->hasher(), leftScalar, lhsTerm->variables());
  } else {
    lhs_new = leftScalar;
  }

  if (rhsTerm) {
    rhs_new = alloc<Term>(rhsTerm->hasher(), rightScalar, rhsTerm->variables());
  } else {
    rhs_new = rightScalar;
  }

  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr PolynomialTransformer::mutate(DivPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Div>(lhs_new, rhs_new));
  }

  // If this is a floating point Div then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Div>(lhs_new, rhs_new);
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

  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr PolynomialTransformer::mutate(ModPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Mod>(lhs_new, rhs_new));
  }

  // 0 % x => 0.
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return lhs_new;
  }

  // x % 1 == 0.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return immLike(v, 0);
  }

  // x % x => 0.
  if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
    return immLike(v, 0);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  if (!lhsTerm) {
    PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
    if (lhsPoly) {
      // Can still optimize this out if we can factorize the polynomial.
      lhsTerm = factorizePolynomial(lhsPoly);
    }
  }

  if (lhsTerm) {
    // ((C1 * C2) * x) % C1 => 0.
    if (rhs_new->isConstant() &&
        immediateEquals(
            evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhs_new)), 0)) {
      return immLike(v, 0);
    }

    // (x * y * z) % x => 0.
    for (const auto& component : lhsTerm->variables()) {
      if (hasher_.hash(component) == hasher_.hash(rhs_new)) {
        return immLike(v, 0);
      }
    }

    // (6 * x * y) % (3 * x * y) => 0.
    // also, (x * y * z) % (z * y) => 0.
    // This requires all variable terms found in the RHS to be present in the
    // LHS.
    TermPtr rhsTerm = to<Term>(rhs_new);
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
              evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhsTerm->scalar())),
              0)) {
        return immLike(v, 0);
      }
    }
  }

  return alloc<Mod>(lhs_new, rhs_new);
}

namespace {

// Combines two MinTerm / MaxTerm expressions into one.
// The first type on the template refers to the op, as in Min or Max and the
// second type refers to the corresponding term, as in MinTerm or MaxTerm.
template <class Op, class OpTerm>
ExprPtr combineMinMaxTerms(
    ExprPtr lhs,
    ExprPtr rhs,
    bool propagate_nans,
    HashProvider& hasher) {
  auto combine_scalars = [&](ExprPtr c1, ExprPtr c2) -> ExprPtr {
    if (c1 && c2) {
      return evaluateOp(alloc<Op>(c1, c2, propagate_nans));
    }
    if (c1) {
      return c1;
    }
    return c2;
  };

  auto combine_opterms = [&](NodePtr<OpTerm> m1, NodePtr<OpTerm> m2) {
    ExprPtr scalar = combine_scalars(m1->scalar(), m2->scalar());
    std::vector<ExprPtr> variables;
    for (const auto& v : m1->variables()) {
      variables.push_back(v);
    }
    for (const auto& v : m2->variables()) {
      variables.push_back(v);
    }
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));
  };

  auto add_expr_to_opterm = [&](ExprPtr expr, NodePtr<OpTerm> opterm) {
    ExprPtr scalar = nullptr;
    std::vector<ExprPtr> variables;
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
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));
  };

  auto lhs_opterm = to<OpTerm>(lhs);
  auto rhs_opterm = to<OpTerm>(rhs);
  if (lhs_opterm && lhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);
  }
  if (rhs_opterm && rhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);
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
    NodePtr<OpTerm> opterm,
    ExprPtr op,
    HashProvider& hasher,
    ExprPtr* other_op) {
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
    ExprPtr lhs,
    ExprPtr rhs,
    bool propagate_nans,
    HashProvider& hasher,
    ExprPtr* new_op) {
  auto lhs_opterm = to<OtherOpTerm>(lhs);
  auto rhs_opterm = to<OtherOpTerm>(rhs);
  if (lhs_opterm && rhs_opterm &&
      lhs_opterm->propagate_nans() == propagate_nans &&
      rhs_opterm->propagate_nans() == propagate_nans) {
    if (!lhs_opterm->scalar() && !rhs_opterm->scalar()) {
      if (lhs_opterm->variables().size() == 2 &&
          rhs_opterm->variables().size() == 2) {
        auto rhs_v1 = rhs_opterm->variables()[0];
        auto rhs_v2 = rhs_opterm->variables()[1];
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ExprPtr new_op_lhs;
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v1, hasher, &new_op_lhs)) {
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v2);
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v1, inner_op);
          return true;
        }
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v2, hasher, &new_op_lhs)) {
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v1);
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v2, inner_op);
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

ExprPtr PolynomialTransformer::mutate(MaxPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Max>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) > 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr new_op;
  if (simplifyNestedMinMax<MaxTerm, MinTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Max, MaxTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

ExprPtr PolynomialTransformer::mutate(MinPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Min>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) < 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr new_op;
  if (simplifyNestedMinMax<MinTerm, MaxTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Min, MinTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

ExprPtr PolynomialTransformer::mutate(CompareSelectPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  ExprPtr true_branch = v->ret_val1()->accept_mutator(this);
  ExprPtr false_branch = v->ret_val2()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant() &&
      true_branch->isConstant() && false_branch->isConstant()) {
    ExprPtr v_new = alloc<CompareSelect>(
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
    return alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
  }

  // If diff is constant, we can determine it.
  ExprPtr diff = alloc<Sub>(rhs_new, lhs_new);
  diff = diff->accept_mutator(this);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  if (!diff->isConstant()) {
    return alloc<CompareSelect>(
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
  return alloc<CompareSelect>(
      lhs_new,
      rhs_new,
      true_branch,
      false_branch,
      v->compare_select_op(),
      v->bias());
}

ExprPtr PolynomialTransformer::mutate(IntrinsicsPtr v) {
  std::vector<ExprPtr> new_params;
  bool changed = false;
  bool allConstant = true;
  for (const auto& p : v->params()) {
    ExprPtr new_child = p->accept_mutator(this);
    new_params.push_back(new_child);

    changed |= p != new_child;
    allConstant &= new_child->isConstant();
  }

  ExprPtr node = v;
  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), new_params);
  }

  if (!allConstant || !v->isPure()) {
    return node;
  }

  // we're evaluating, but the evaluator only supports float intrinsics.
  std::vector<ExprPtr> const_params;
  changed = false;
  for (const auto& p : new_params) {
    if (p->dtype().scalar_type() == ScalarType::Float) {
      const_params.push_back(p);
    } else {
      const_params.push_back(
          alloc<Cast>(Dtype(ScalarType::Float, p->dtype().lanes()), p));
      changed = true;
    }
  }

  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), const_params);
  }
  return evaluateOp(node);
}

ExprPtr PolynomialTransformer::mutate(CastPtr v) {
  ExprPtr node = v->src_value()->accept_mutator(this);
  if (node->isConstant()) {
    return evaluateOp(alloc<Cast>(v->dtype(), node));
  }

  if (v->dtype() == node->dtype()) {
    return node;
  }

  return alloc<Cast>(v->dtype(), node);
}

ExprPtr PolynomialTransformer::mutate(IfThenElsePtr v) {
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

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

  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr PolynomialTransformer::mutate(AndPtr v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(XorPtr v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(LshiftPtr v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(RshiftPtr v) {
  return mutateBinaryOp(v, this);
}

StmtPtr PolynomialBase::mutate(CondPtr v) {
  ExprPtr cond_old = v->condition();
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  ExprPtr cond_new = cond_old->accept_mutator(this);
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  // If the condition is constant then we can choose the right branch now.
  if (cond_new->isConstant()) {
    if (!immediateEquals(cond_new, 0)) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return true_new;
    } else {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      return false_new;
    }
  }

  // If both branches are the same then don't do the condition.
  if (true_new && false_new &&
      hasher_.hash(true_new) == hasher_.hash(false_new)) {
    return true_new;
  }

  BlockPtr true_block = to<Block>(true_new);
  BlockPtr false_block = to<Block>(false_new);
  bool true_empty = !true_new || (true_block && true_block->nstmts() == 0);
  bool false_empty = !false_new || (false_block && false_block->nstmts() == 0);

  if (true_empty && false_empty) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }
  if (cond_old != cond_new) {
    v->set_condition(cond_new);
  }
  if (true_old != true_new) {
    v->set_true_stmt(true_new);
  }
  if (false_old != false_new) {
    v->set_false_stmt(false_new);
  }
  return v;
}

static StmtPtr handleForCondReordering(ForPtr loop, CondPtr cond) {
  if (cond->false_stmt()) {
    return nullptr;
  }

  auto condition_vars = VarFinder::find(cond->condition());
  for (const auto& v : condition_vars) {
    // If the condition depends on a Var that is modified in the loop body, it
    // may not be safe to reorder.
    if (ModifiesVarChecker::check(loop, v)) {
      return nullptr;
    }
  }

  ForPtr new_f = loop->cloneWithNewBody(Stmt::clone(cond->true_stmt()));
  return cond->cloneWithNewBody(new_f);
}

StmtPtr PolynomialBase::mutate(ForPtr v) {
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body;

  ExprPtr loops = alloc<Sub>(stop_new, start_new);
  loops = loops->accept_mutator(this);
  if (loop_options.isDefault() && loops->isConstant()) {
    if (immediateEquals(loops, 0)) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    } else if (immediateEquals(loops, 1)) {
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);
      return body_new;
    }
  }

  body_new = body_new->accept_mutator(this);
  if (!body_new) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }

  if (auto block = to<Block>(body_new)) {
    if (block->nstmts() == 0) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    }

    if (block->nstmts() == 1) {
      if (auto cond = to<Cond>(block->front())) {
        StmtPtr reordered = handleForCondReordering(v, cond);
        if (reordered) {
          return reordered->accept_mutator(this);
        }
      }
    }
  }

  if (var != var_new) {
    v->set_var(var_new);
  }
  if (start != start_new) {
    v->set_start(start_new);
  }
  if (stop != stop_new) {
    v->set_stop(stop_new);
  }
  if (body != body_new) {
    v->set_body(body_new);
  }
  return v;
}

StmtPtr PolynomialBase::mutate(BlockPtr v) {
  std::vector<StmtPtr> stmts;
  // Flatten sub-blocks:
  bool stmts_changed = false;
  for (const StmtPtr& stmt : *v) {
    StmtPtr stmt_new = stmt->accept_mutator(this);
    stmts_changed |= stmt != stmt_new;
    if (stmt_new == nullptr) {
      continue;
    }

    if (auto subBlock = to<Block>(stmt_new)) {
      for (Block::iterator I = subBlock->begin(), E = subBlock->end();
           I != E;) {
        // Be careful to avoid invalidating the iterator.
        StmtPtr s = *(I++);
        subBlock->remove_stmt(s);
        stmts.push_back(s);
      }
      stmts_changed = true;
    } else {
      stmts.push_back(stmt_new);
    }
  }
  if (stmts_changed) {
    v->set_stmts(stmts);
  }
  return v;
}

// TermExpander

ExprPtr TermExpander::mutate(TermPtr v) {
  ExprPtr newScalar = v->scalar()->accept_mutator(this);
  if (immediateEquals(newScalar, 0)) {
    return newScalar;
  }

  std::vector<ExprPtr> vars;
  std::vector<ExprPtr> multilaneVars;

  // Assume we can reorder here because we wont merge floating terms.
  ExprPtr lastNode{nullptr};
  for (const auto& var : v->variables()) {
    ExprPtr node = var->accept_mutator(this);
    if (MulPtr mul = to<Mul>(node)) {
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

  for (const auto& node : multilaneVars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      lastNode = mulMultilane(lastNode, node);
      // simplify first, then re-expand.
      lastNode = lastNode->accept_mutator(simplifier_);
      lastNode = lastNode->accept_mutator(this);
    }
  }

  for (const auto& node : vars) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      lastNode = alloc<Mul>(lastNode, node);
    }
  }

  if (!immediateEquals(newScalar, 1)) {
    if (lastNode) {
      // We want to avoid a leaving a CastNode on the scalar, so handle that
      // now.
      auto termDtype = v->scalar()->dtype();
      auto lastNodeDtype = lastNode->dtype();
      if (termDtype != lastNodeDtype) {
        ExprPtr castV = v->scalar();
        // Take care of lane mismatch first.
        if (termDtype.lanes() != lastNodeDtype.lanes()) {
          castV = alloc<Broadcast>(v->scalar(), lastNodeDtype.lanes());
        }
        // Now take care of scalar type as well.
        if (termDtype.scalar_type() != lastNodeDtype.scalar_type()) {
          castV = alloc<Cast>(lastNode->dtype(), castV);
          // For scalars, we can simplify the cast further.
          if (lastNodeDtype.lanes() == 1) {
            castV = evaluateOp(castV);
          }
        }
        lastNode = alloc<Mul>(castV, lastNode);
      } else {
        lastNode = alloc<Mul>(v->scalar(), lastNode);
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
static ExprPtr polyGCD(PolynomialPtr poly) {
  ExprPtr scalar = poly->scalar();
  const std::vector<TermPtr>& variables = poly->variables();

  // We ony want to factorize if we're saving complete operations, i.e. no
  // value in factorizing 6x + 4y into 2 * (3x + 2y) since we don't save work.
  int opsSaved = 1; // default to saving the scalar.
  long GCD = std::abs(immediateAs<long>(scalar));
  for (const auto& t : variables) {
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

  return immLike(poly, GCD);
}

// A ModRound is a div-mod-mul in which the divisor in div and multiplier in mul
// are identical and not equal to 1.
// In a ModRound x/y%z*y*c (c is constant), 'scalar' denotes c, 'denominator'
// denotes x, 'divisor' denotes y and 'mod_divisor' denotes z.
class ModRound {
 public:
  ModRound(ExprPtr scalar, ExprPtr denom, ExprPtr divisor, ExprPtr mod_divisor)
      : scalar(std::move(scalar)),
        denom(std::move(denom)),
        divisor(std::move(divisor)),
        mod_divisor(std::move(mod_divisor)) {}
  ExprPtr scalar;
  ExprPtr denom;
  ExprPtr divisor;
  ExprPtr mod_divisor;
};

static std::optional<class ModRound> isModRound(TermPtr e) {
  DivPtr div{nullptr};
  ModPtr mod{nullptr};
  ExprPtr denom{nullptr};
  ExprPtr divisor{nullptr};
  ExprPtr mod_divisor{nullptr};
  ExprPtr multiplier = e->scalar();
  ExprPtr scalar{nullptr};
  ExprPtr other{nullptr};

  for (const auto& m : e->variables()) {
    if (m->expr_type() == IRNodeType::kMod) {
      // TODO: currently only identify terms with one variable being mod; it is
      // possible to extend this if we have to handle terms like (t/(x%2 * y) %
      // z) * (x%2 *y).
      if (!mod) {
        mod = to<Mod>(m);
      } else {
        return c10::nullopt;
      }
    } else {
      // Take care of special cases before multiplying the scalar and variable.
      if (multiplier->isConstant()) {
        // Take care of lane mismatch first.
        if (multiplier->dtype().lanes() != m->dtype().lanes()) {
          multiplier = alloc<Broadcast>(multiplier, m->dtype().lanes());
        }
        // Take care of scalar type mismatch.
        if (multiplier->dtype().scalar_type() != m->dtype().scalar_type()) {
          multiplier = alloc<Cast>(m->dtype(), multiplier);
          if (m->dtype().lanes() == 1) {
            multiplier = evaluateOp(multiplier);
          }
        }
      }

      // All non-mod variables are considered as part of the multiplier.
      multiplier = alloc<Mul>(multiplier, m);
    }
  }
  multiplier = IRSimplifier::simplify(multiplier);

  if (!mod) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return c10::nullopt;
  }

  mod_divisor = IRSimplifier::simplify(mod->rhs());
  other = mod->lhs();

  if (!(div = to<Div>(other))) {
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
      if (immediateEquals(evaluateOp(alloc<Mod>(multiplier, divisor)), 0)) {
        // The common factor becomes 'scalar' of the term, e.g.,in t/3%7*6,
        // divisor=multiplier=3, scalar=2.
        ExprPtr c = evaluateOp(alloc<Div>(multiplier, divisor));
        scalar = c;
      } else if (immediateEquals(
                     evaluateOp(alloc<Mod>(divisor, multiplier)), 0)) {
        // The common factor becomes part of 'denom', e.g., in t/14%7*2,
        // divisor=multiplier=2, denom=t/7.
        ExprPtr c = evaluateOp(alloc<Div>(divisor, multiplier));
        divisor = multiplier;
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        denom = IRSimplifier::simplify(alloc<Div>(other, c));
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
    scalar = immLike(multiplier, 1);
  }

  return ModRound(scalar, denom, divisor, mod_divisor);
}

// Search the polynomial for Terms that can be merged in
// (1) Round + Mod pattern: (x/y) * y + x % y => RoundOff(x,y) + Mod(x, y) => x
// (2) Mod round + Mod pattern: (x/y % z)*y + x%y => ModRound(x, y, z) + Mod(x,
// y) => x % (y*z)
static ExprPtr simplifyRoundModPattern(PolynomialPtr poly) {
  std::vector<TermPtr> rounds;
  std::vector<TermPtr> mods;
  std::vector<TermPtr> mod_rounds;
  std::vector<TermPtr> others;

  // Split out the Mod, ModRounds and RoundOffs operations so we can inspect.
  for (const auto& c : poly->variables()) {
    if (c->variables().size() > 1) {
      if (auto a = isModRound(c)) {
        mod_rounds.push_back(c);
      } else {
        others.push_back(c);
      }
      continue;
    }

    ExprPtr e = c->variables()[0];

    if (to<RoundOff>(e)) {
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
  std::vector<TermPtr> mods_merged;
  bool repeat = true;
  // Repeat merging terms till there are no Mods or the terms cannot be merged
  // any further.
  while (!mods.empty() && repeat) {
    repeat = false;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (int64_t i = mods.size() - 1; i >= 0; i--) {
      TermPtr m = mods[i];
      ModPtr mod = to<Mod>(m->variables()[0]);
      CHECK(mod);
      ExprPtr mod_lhs = IRSimplifier::simplify(mod->lhs());
      ExprPtr mod_rhs = IRSimplifier::simplify(mod->rhs());
      bool merged = false;
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      for (int64_t j = mod_rounds.size() - 1; j >= 0; j--) {
        TermPtr mr = mod_rounds[j];
        auto a = isModRound(mr);
        CHECK(a);
        ModRound& mod_round = *a;

        // TODO: for now don't attempt partial factorization of this
        // optimization. E.g. it's possible to do: 2 * (x/y%z) * y + (x%y) =>
        // x%(y*z) + (x/y%z) * y
        if (!immediateEquals(
                evaluateOp(alloc<Sub>(mod_round.scalar, m->scalar())), 0)) {
          continue;
        }
        // Valid optimization if mod LHS matches denom and mod RHS matches
        // divisor.
        if (hasher.hash(mod_round.denom) == hasher.hash(mod_lhs) &&
            hasher.hash(mod_round.divisor) == hasher.hash(mod_rhs)) {
          // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
          TermPtr merged_m = alloc<Term>(
              hasher,
              mod_round.scalar,
              IRSimplifier::simplify(alloc<Mod>(
                  mod_round.denom,
                  alloc<Mul>(mod_round.divisor, mod_round.mod_divisor))));
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
      for (int64_t k = rounds.size() - 1; k >= 0; k--) {
        TermPtr r = rounds[k];
        RoundOffPtr roundoff = to<RoundOff>(r->variables()[0]);
        CHECK(roundoff);

        // TODO: for now don't attempt partial factorization of this
        // optimization. E.g. it's possible to do: 2 * (x/y) * y + (x%y) => x +
        // (x/y) * y but unsure thats actually much better, particularly with
        // CSE.
        if (!immediateEquals(
                evaluateOp(alloc<Sub>(r->scalar(), m->scalar())), 0)) {
          continue;
        }
        ExprPtr round_lhs = IRSimplifier::simplify(roundoff->lhs());
        ExprPtr round_rhs = IRSimplifier::simplify(roundoff->rhs());
        // Valid optimization if LHS and RHS are equal for both.
        if (hasher.hash(round_lhs) == hasher.hash(mod_lhs) &&
            hasher.hash(round_rhs) == hasher.hash(mod_rhs)) {
          TermPtr merged_r = alloc<Term>(hasher, r->scalar(), round_lhs);
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

  return alloc<Polynomial>(hasher, poly->scalar(), others);
}

// Trivially factorize terms by GCD of scalar components.
TermPtr PolynomialBase::factorizePolynomial(PolynomialPtr poly) {
  ExprPtr scalar = poly->scalar();
  const std::vector<TermPtr>& variables = poly->variables();

  // Compute the GCD of terms.
  ExprPtr GCD = polyGCD(poly);

  // No GCD means 0 or 1 and can't be factored.
  if (!GCD) {
    return nullptr;
  }

  // Create new structure.
  std::vector<TermPtr> newPolyTerms;
  newPolyTerms.reserve(variables.size());
  for (const auto& t : variables) {
    // New term with the scalar divided by the GCD.
    newPolyTerms.push_back(alloc<Term>(
        poly->hasher(),
        evaluateOp(alloc<Div>(t->scalar(), GCD)),
        t->variables()));
  }

  PolynomialPtr newPoly = alloc<Polynomial>(
      poly->hasher(), evaluateOp(alloc<Div>(scalar, GCD)), newPolyTerms);

  return alloc<Term>(poly->hasher(), GCD, newPoly);
}

ExprPtr TermExpander::mutate(PolynomialPtr v) {
  if (v->variables().empty()) {
    return v->scalar();
  }

  // If this Polynomial can be factorized: do it, then expand the result.
  if (ExprPtr simplified = simplifyRoundModPattern(v)) {
    return simplified->accept_mutator(this);
  }

  // If this Polynomial can be factorized: do it, then expand the result.
  if (ExprPtr factorized = factorizePolynomial(v)) {
    return factorized->accept_mutator(this);
  }

  std::vector<TermPtr> addTerms;
  std::vector<TermPtr> subTerms;

  auto vars = v->variables();
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(vars.begin(), vars.end(), [&](ExprPtr a, ExprPtr b) {
    if (!str_repr_cache.count(a)) {
      str_repr_cache[a] = std::to_string(a);
    }
    if (!str_repr_cache.count(b)) {
      str_repr_cache[b] = std::to_string(b);
    }
    return str_repr_cache.at(a) < str_repr_cache.at(b);
  });

  // partition the terms into a list to add and list to subtract.
  for (const auto& node : vars) {
    if (immediateIsNegative(node->scalar())) {
      subTerms.push_back(node);
    } else if (!immediateEquals(node->scalar(), 0)) {
      addTerms.push_back(node);
    }
    // Skip terms with a scalar of zero.
  }

  // The last node constructed.
  ExprPtr lastNode{nullptr};

  for (const auto& node : addTerms) {
    ExprPtr simpleNode = node->accept_mutator(this);

    if (lastNode == nullptr) {
      lastNode = simpleNode;
      continue;
    }

    if (isMultilanePrimitive(simpleNode)) {
      auto ret = combineMultilane<Add>(lastNode, simpleNode);
      if (ret) {
        // simplify result first, then expand.
        lastNode = ret->accept_mutator(simplifier_);
        lastNode = lastNode->accept_mutator(this);
        continue;
      }
    }

    lastNode = alloc<Add>(lastNode, simpleNode);
  }

  // If we have no add terms the scalar should go first.
  // E.g. 1 - x.
  bool scalarWritten = false;
  if (lastNode == nullptr) {
    auto scalarNode = v->scalar()->accept_mutator(simplifier_);

    if (!immediateEquals(scalarNode, 0)) {
      lastNode = scalarNode;
      scalarWritten = true;
    }
  }

  for (const auto& node : subTerms) {
    // Can still be first node if scalarVal is 0.
    if (lastNode == nullptr) {
      lastNode = node->accept_mutator(this);
      continue;
    }

    // Negate the term back to positive since we'll be subtracting it.
    ExprPtr negated =
        evaluateOp(alloc<Mul>(immLike(node->scalar(), -1), node->scalar()));
    TermPtr newRHS = alloc<Term>(node->hasher(), negated, node->variables());
    lastNode = alloc<Sub>(lastNode, newRHS->accept_mutator(this));
  }

  if (scalarWritten || immediateEquals(v->scalar(), 0)) {
    if (!lastNode) {
      return immLike(v, 0);
    }
    return lastNode;
  }

  if (immediateIsNegative(v->scalar())) {
    // Negate the scalar and subtract.
    ExprPtr negated =
        evaluateOp(alloc<Mul>(immLike(lastNode, -1), v->scalar()));
    lastNode = alloc<Sub>(lastNode, evaluateOp(negated));
  } else {
    // we want to avoid a cast to the scalar if it would happen.
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (v->scalar()->dtype() != lastNode->dtype()) {
      lastNode = alloc<Add>(
          lastNode, evaluateOp(alloc<Cast>(lastNode->dtype(), v->scalar())));
    } else {
      lastNode = alloc<Add>(lastNode, v->scalar());
    }
  }

  return lastNode;
}

ExprPtr TermExpander::mutate(MaxTermPtr v) {
  auto& variables = v->variables();
  if (variables.empty()) {
    if (!v->scalar()) {
      // This case should never happen because MaxTerm will be created only
      // on valid Max expressions.
      throw std::logic_error("empty maxterm op");
    }
    return v->scalar();
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr max;
  if (v->scalar()) {
    max = alloc<Max>(variables[0], v->scalar(), v->propagate_nans());
  } else {
    max = variables[0];
  }
  for (size_t i = 1; i < variables.size(); i++) {
    max = alloc<Max>(max, variables[i], v->propagate_nans());
  }
  return max->accept_mutator(this);
}

ExprPtr TermExpander::mutate(MinTermPtr v) {
  auto& variables = v->variables();
  if (variables.empty()) {
    if (!v->scalar()) {
      // This case should never happen because MinTerm will be created only
      // on valid Min expressions.
      throw std::logic_error("empty minterm op");
    }
    return v->scalar();
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr min;
  if (v->scalar()) {
    min = alloc<Min>(variables[0], v->scalar(), v->propagate_nans());
  } else {
    min = variables[0];
  }
  for (size_t i = 1; i < variables.size(); i++) {
    min = alloc<Min>(min, variables[i], v->propagate_nans());
  }
  return min->accept_mutator(this);
}

// Expands RoundOff(x, y) => Term(1, Div(x, y), y), which will later be expanded
// to Mul(Div(x, y), y).
ExprPtr TermExpander::mutate(RoundOffPtr v) {
  TermPtr term = alloc<Term>(
      simplifier_->hasher(),
      immLike(v, 1),
      alloc<Div>(v->lhs(), v->rhs()),
      v->rhs());
  return term->accept_mutator(this);
}

ExprPtr buf_flat_size(BufPtr v) {
  std::vector<ExprPtr> dims = v->dims();
  if (dims.empty()) {
    return alloc<LongImm>(1);
  }
  ExprPtr flattened = immLike(dims[0], 1);
  for (auto& dim : dims) {
    flattened = alloc<Mul>(flattened, dim);
  }
  flattened = IRSimplifier::simplify(flattened);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return flattened;
}

StmtPtr TermExpander::mutate(AllocatePtr v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new,
      buildErrorMessage("TermExpander mutation produced null for Buf."));
  ExprPtr flattened = buf_flat_size(buf_new);

  if (flattened->isConstant() && immediateEquals(flattened, 0)) {
    eliminated_allocations_.insert(buf_new->base_handle());
    return nullptr;
  }

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

StmtPtr TermExpander::mutate(FreePtr v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new,
      buildErrorMessage("TermExpander mutation produced null for Buf."));

  if (eliminated_allocations_.count(buf_new->base_handle())) {
    eliminated_allocations_.erase(buf_new->base_handle());
    return nullptr;
  }

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

// Combines adjacent Cond nodes with identical conditions.
BlockPtr TermExpander::fuseConditions(BlockPtr v) {
  std::vector<StmtPtr> stmts;
  bool did_anything = false;
  CondPtr prev_cond = nullptr;

  for (const auto& s : *v) {
    CondPtr cond = to<Cond>(s);
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
    BlockPtr true_block = alloc<Block>(std::vector<StmtPtr>({}));
    BlockPtr false_block = alloc<Block>(std::vector<StmtPtr>({}));

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
    StmtPtr new_cond = prev_cond->cloneWithNewBodies(true_block, false_block)
                           ->accept_mutator(this);
    prev_cond = to<Cond>(new_cond);

    // erase, which shortens the list.
    stmts.pop_back();
    stmts.push_back(new_cond);
    did_anything = true;
  }

  if (!did_anything) {
    return v;
  }

  // clean up parents.
  for (const auto& s : stmts) {
    if (s->get_parent() == v) {
      v->remove_stmt(s);
    }
  }

  return alloc<Block>(stmts);
}

StmtPtr TermExpander::fuseSyncThreads(BlockPtr block) {
  // only really first if highest level Block.
  bool first = block->get_parent() == nullptr;
  SyncThreadsPtr last = nullptr;
  std::vector<StmtPtr> stmts;
  bool did_anything = false;

  for (const auto& s : *block) {
    SyncThreadsPtr sync = to<SyncThreads>(s);
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
  for (const auto& s : stmts) {
    if (s->get_parent() == block) {
      block->remove_stmt(s);
    }
  }

  return alloc<Block>(std::vector<StmtPtr>({stmts}));
}

StmtPtr TermExpander::mutate(BlockPtr v) {
  StmtPtr new_stmt = PolynomialBase::mutate(v);
  BlockPtr new_block = to<Block>(new_stmt);
  if (!new_block) {
    return new_stmt;
  }

  // fuseConditions will return the original block if it cannot fuse.
  new_block = fuseConditions(new_block);
  /// fuseSyncThreads too.
  return fuseSyncThreads(new_block);
}

// SimplifierUnderContext
//
// This function records the bounds(range) info of the index var in a for-stmt.
// The bounds info will be used later when simplifying expressions with the
// index var.
StmtPtr SimplifierUnderContext::mutate(ForPtr v) {
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body;

  // save bounds info before this for-stmt
  //
  // The same variable could have appeared in a if-stmt which the for-stmt is
  // nested inside, and we need to restore its bounds info after the for-stmt.
  //
  // An example,
  // if (i>=0 && i<5) {
  //   for (i=0; i<3; i++){
  //     A[i] = ...
  //   }
  //   x = (i+20) / 5;
  //}
  // Inside the if stmt, i is in the range of [0, 5); and if we can restore this
  // bound info after the for stmt, we can use it to simplify the assignment
  // stmt x = (i+20)/5 to x = 4.
  bool has_bounds = false;
  analysis::Bound bound_old;
  VarPtr var_key = to<Var>(var);
  auto got = var_bound_info_.find(var_key);
  if (got != var_bound_info_.end()) {
    has_bounds = true;
    bound_old = got->second;
  }
  // set bounds info for index var
  const analysis::Bound bound_new(start_new, stop_new);
  var_bound_info_[var_key] = bound_new;

  ExprPtr iters = alloc<Sub>(stop_new, start_new);
  iters = iters->accept_mutator(this);
  if (loop_options.isDefault() && iters->isConstant()) {
    if (immediateEquals(iters, 0)) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    } else if (immediateEquals(iters, 1)) {
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);

      // erase index var bounds info or restore old bounds info
      if (has_bounds) {
        var_bound_info_[var_key] = bound_old;
      } else {
        var_bound_info_.erase(var_key);
      }

      return body_new;
    }
  }

  body_new = body_new->accept_mutator(this);

  // erase index var bounds info or restore old bounds info
  if (has_bounds) {
    var_bound_info_[var_key] = bound_old;
  } else {
    var_bound_info_.erase(var_key);
  }

  if (!body_new) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }

  if (auto block = to<Block>(body_new)) {
    if (block->nstmts() == 0) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    }

    if (block->nstmts() == 1) {
      // if the stmt in the loop body is a if-stmt, try to move the branching
      // out of the loop
      if (auto cond = to<Cond>(block->front())) {
        StmtPtr reordered = handleForCondReordering(v, cond);
        if (reordered) {
          return reordered->accept_mutator(this);
        }
      }
    }
  }

  if (var != var_new) {
    v->set_var(var_new);
  }
  if (start != start_new) {
    v->set_start(start_new);
  }
  if (stop != stop_new) {
    v->set_stop(stop_new);
  }
  if (body != body_new) {
    v->set_body(body_new);
  }
  return v;
}

// Simplify division using distributive laws for the following cases:
// 1) (i + x) / n => x/n, if
//   a) n is a positive integer constant;
//   b) i is the index var of a for-stmt and the range of i is
// a subset of [0, n);
//   c) x is a constant and the end value of i's range is less than n - x%n;
//   TODO: remove d) from the requirements because the simplification formula
//   still holds when x is a negative integer. In integer division, the result
//   of the division is converted to an integer using `floor` function which
//   returns the largest integer that is not greater than X. For example, -1/6
//   returns -1. But currently, both Pytorch and NNC are performing an incorrect
//   integer division: (-1)/6 = 0. With the current implementation of integer
//   division, x has to be not negative. d) x is not negative
//
// 2) (i + j*n) / n => j, if
//   a) n is a positive integer constant;
//   b) i is the index var of a for-stmt and the range of i is
// a subset of [0, n);
//   c) j is an integer variable;
//   TODO: remove d) from the requirements because the simplification formula
//   still holds when j is a negative integer. In integer division, the result
//   of the division is converted to an integer using `floor` function which
//   returns the largest integer that is not greater than X. For example, -1/6
//   returns -1. But currently, both Pytorch and NNC are performing an incorrect
//   integer division: (-1)/6 = 0. With the current implementation of integer
//   division, x has to be not negative. d) j is not negative
static ExprPtr distributeDiv(
    ExprPtr lhs,
    ExprPtr rhs,
    VarBoundInfo var_bound_info) {
  if (!lhs || !rhs) {
    return nullptr;
  }
  // return if not integer division
  if (lhs->dtype().is_floating_point() || rhs->dtype().is_floating_point()) {
    return nullptr;
  }

  // identify n: a positive integer constant
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (!rhsScalar) {
    return nullptr;
  }
  ExprPtr check_n_value = IRSimplifier::simplify(
      alloc<CompareSelect>(rhsScalar, immLike(rhsScalar, 0), kGT));
  if (!immediateEquals(check_n_value, 1)) {
    return nullptr;
  }

  auto lhsAdd = to<Add>(lhs);
  if (!lhsAdd) {
    return nullptr;
  }
  ExprPtr lhsAdd1 = lhsAdd->lhs();
  ExprPtr lhsAdd2 = lhsAdd->rhs();

  // identify index var 'i'
  VarPtr var_key = to<Var>(lhsAdd1);
  ExprPtr main = lhsAdd2;
  if (var_key == nullptr) {
    var_key = to<Var>(lhsAdd2);
    main = lhsAdd1;
  }

  if (var_key == nullptr) {
    return nullptr;
  }

  auto got = var_bound_info.find(var_key);
  if (got == var_bound_info.end()) {
    return nullptr;
  }

  // check the bounds of 'i'
  auto start = got->second.start;
  // open upper bound, i.e.,  end is one more than the maximum value in the
  // range
  auto end = got->second.end;
  ExprPtr check_start = IRSimplifier::simplify(
      alloc<CompareSelect>(start, immLike(start, 0), kGE));
  ExprPtr check_end =
      IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
  if (!check_start->isConstant() || !check_end->isConstant() ||
      !immediateEquals(check_start, 1) || !immediateEquals(check_end, 1)) {
    return nullptr;
  }

  ExprPtr ret = IRSimplifier::simplify(alloc<Div>(main, rhsScalar));

  // simplify type 1) exprs: '(i+x)/n' => 'x/n'
  ExprPtr sign_check =
      IRSimplifier::simplify(alloc<CompareSelect>(main, immLike(main, 0), kGE));
  ExprPtr main_mod = IRSimplifier::simplify(alloc<Mod>(main, rhsScalar));
  ExprPtr mod_check = IRSimplifier::simplify(
      alloc<CompareSelect>(alloc<Add>(main_mod, end), rhsScalar, kLE));
  if (sign_check->isConstant() && immediateEquals(sign_check, 1) &&
      mod_check->isConstant() && immediateEquals(mod_check, 1)) {
    return ret;
  }

  // simplify type 2 exprs: '(i+j*n)/n' => 'j'
  auto ret_var = to<Var>(ret);
  // FIXME: Allow any integral type.
  if (ret_var && ret_var->dtype() == kInt) {
    // retrieve j's range info
    auto got = var_bound_info.find(ret_var);
    if (got == var_bound_info.end()) {
      return nullptr;
    }

    // check if j is not negative
    sign_check = IRSimplifier::simplify(alloc<CompareSelect>(
        got->second.start, immLike(got->second.start, 0), kGE));
    if (sign_check->isConstant() && immediateEquals(sign_check, 1)) {
      return ret_var;
    }
  }

  return nullptr;
}

// Simplify mod using distributive laws for the following cases:
// 1) (i + x) % n => i + x%n if
//   a) n is a positive integer constant;
//   b) i is the index var of a for-stmt and the range of i is
// a subset of [0, n);
//   c) x is a constant and the end value of i's range is less than n - x%n;
//   TODO: remove d) from the requirements because the simplification formula
//   still holds when x is a negative integer. In integer division, the result
//   of the division is converted to an integer using `floor` function which
//   returns the largest integer that is not greater than X. For example, -1/6
//   returns -1. But currently, both Pytorch and NNC are performing an incorrect
//   integer division: (-1)/6 = 0. With the current implementation of integer
//   division, x has to be not negative. d) x is not negative
//
// 2) (i + j*n) % n => i if
//   a) n is a positive integer constant;
//   b) i is the index var of a for-stmt and the range of i is
// a subset of [0, n);
//   c) j is an integer variable;
//   TODO: remove d) from the requirements because the simplification formula
//   still holds when j is a negative integer. In integer division, the result
//   of the division is converted to an integer using `floor` function which
//   returns the largest integer that is not greater than X. For example, -1/6
//   returns -1. But currently, both Pytorch and NNC are performing an incorrect
//   integer division: (-1)/6 = 0. With the current implementation of integer
//   division, j has to be not negative. d) j is not negative
static ExprPtr distributeMod(
    ExprPtr lhs,
    ExprPtr rhs,
    VarBoundInfo var_bound_info) {
  if (!lhs || !rhs) {
    return nullptr;
  }
  // return if not integer mod
  if (lhs->dtype().is_floating_point() || rhs->dtype().is_floating_point()) {
    return nullptr;
  }

  // identify n: a positive integer constant
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (!rhsScalar) {
    return nullptr;
  }
  ExprPtr check_n_value = IRSimplifier::simplify(
      alloc<CompareSelect>(rhsScalar, immLike(rhsScalar, 0), kGT));
  if (!immediateEquals(check_n_value, 1)) {
    return nullptr;
  }

  auto lhsAdd = to<Add>(lhs);
  if (!lhsAdd) {
    return nullptr;
  }
  if (!lhsAdd || !rhsScalar) {
    return nullptr;
  }
  ExprPtr lhsAdd1 = lhsAdd->lhs();
  ExprPtr lhsAdd2 = lhsAdd->rhs();

  // identify index var 'i'
  VarPtr var_key = to<Var>(lhsAdd1);
  ExprPtr main = lhsAdd2;
  if (var_key == nullptr) {
    var_key = to<Var>(lhsAdd2);
    main = lhsAdd1;
  }
  if (var_key == nullptr) {
    return nullptr;
  }

  auto got = var_bound_info.find(var_key);
  if (got == var_bound_info.end()) {
    return nullptr;
  }

  // check the bounds of 'i'
  auto start = got->second.start;
  // open upper bound, i.e.,  end is one more than the maximum value in the
  // range
  auto end = got->second.end;
  ExprPtr check_start = IRSimplifier::simplify(
      alloc<CompareSelect>(start, immLike(start, 0), kGE));
  ExprPtr check_end =
      IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
  if (!check_start->isConstant() || !check_end->isConstant() ||
      !immediateEquals(check_start, 1) || !immediateEquals(check_end, 1)) {
    return nullptr;
  }

  // simplify type 1) exprs: '(i+x)%n' => 'i+x%n'
  ExprPtr sign_check =
      IRSimplifier::simplify(alloc<CompareSelect>(main, immLike(main, 0), kGE));
  ExprPtr main_mod = IRSimplifier::simplify(alloc<Mod>(main, rhsScalar));
  ExprPtr mod_check = IRSimplifier::simplify(
      alloc<CompareSelect>(alloc<Add>(main_mod, end), rhsScalar, kLE));
  if (sign_check->isConstant() && immediateEquals(sign_check, 1) &&
      mod_check->isConstant() && immediateEquals(mod_check, 1)) {
    return alloc<Add>(var_key, main_mod);
  }

  // simplify type 2) exprs: '(i+j*n)%n' => 'i'
  ExprPtr main_div = IRSimplifier::simplify(alloc<Div>(main, rhsScalar));
  auto j_var = to<Var>(main_div);
  // FIXME: Allow any integral type.
  if (j_var && j_var->dtype() == kInt) {
    // retrieve j's range info
    auto got = var_bound_info.find(j_var);
    if (got == var_bound_info.end()) {
      return nullptr;
    }

    // check if j is not negative
    sign_check = IRSimplifier::simplify(alloc<CompareSelect>(
        got->second.start, immLike(got->second.start, 0), kGE));
    if (sign_check->isConstant() && immediateEquals(sign_check, 1)) {
      return var_key;
    }
  }

  return nullptr;
}

ExprPtr SimplifierUnderContext::mutate(DivPtr v) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();

  std::ostringstream oss;
  if (auto ret = distributeDiv(lhs, rhs, var_bound_info_)) {
    GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *ret);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return ret->accept_mutator(this);
  }

  // i / N -> 0 if the range of i's values is a subset of [0, N)
  // where N is an integer constant
  auto lhsVar = to<Var>(lhs);
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (lhsVar && rhsScalar && !rhsScalar->dtype().is_floating_point()) {
    auto got = var_bound_info_.find(lhsVar);
    if (got != var_bound_info_.end()) {
      auto start = got->second.start;
      auto end = got->second.end;
      ExprPtr check_start = IRSimplifier::simplify(
          alloc<CompareSelect>(start, immLike(start, 0), kGE));
      ExprPtr check_end =
          IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
      if (check_start->isConstant() && check_end->isConstant() &&
          immediateEquals(check_start, 1) && immediateEquals(check_end, 1)) {
        GRAPH_DEBUG(
            "SimplifierUnderContext: ", *v, " => ", *immLike(lhsVar, 0));
        return immLike(lhsVar, 0);
      }
    }
  }

  ExprPtr lhs_new = lhs->accept_mutator(this);
  ExprPtr rhs_new = rhs->accept_mutator(this);
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr SimplifierUnderContext::mutate(IfThenElsePtr v) {
  ExprPtr condition = v->condition();
  ExprPtr true_val = v->true_value();
  ExprPtr false_val = v->false_value();

  auto simplified_condition =
      IRSimplifier::simplify(condition->accept_mutator(this));
  auto simplified_true_val =
      IRSimplifier::simplify(true_val->accept_mutator(this));
  auto simplified_false_val =
      IRSimplifier::simplify(false_val->accept_mutator(this));
  if (simplified_condition->isConstant()) {
    return immediateAs<int>(simplified_condition) ? simplified_true_val
                                                  : simplified_false_val;
  }

  bool nothing_changed = (simplified_condition == condition) &&
      (simplified_true_val == true_val) && (simplified_false_val == false_val);
  return nothing_changed
      ? v
      : alloc<IfThenElse>(
            simplified_condition, simplified_true_val, simplified_false_val);
}

ExprPtr SimplifierUnderContext::mutate(CompareSelectPtr v) {
  GRAPH_DEBUG("(SimplifierUnderContext) Original: ", std::to_string(v));

  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr ret1 = v->ret_val1();
  ExprPtr ret2 = v->ret_val2();

  auto simplified_lhs = IRSimplifier::simplify(lhs->accept_mutator(this));
  auto simplified_rhs = IRSimplifier::simplify(rhs->accept_mutator(this));
  auto simplified_ret1 = IRSimplifier::simplify(ret1->accept_mutator(this));
  auto simplified_ret2 = IRSimplifier::simplify(ret2->accept_mutator(this));

  ExprPtr simplified_cmp_select_expr = nullptr;
  if ((simplified_lhs == lhs) && (simplified_rhs == rhs) &&
      (simplified_ret1 == ret1) && (simplified_ret2 == ret2)) {
    simplified_cmp_select_expr = v;
  } else {
    simplified_cmp_select_expr = alloc<CompareSelect>(
        simplified_lhs,
        simplified_rhs,
        simplified_ret1,
        simplified_ret2,
        v->compare_select_op(),
        v->bias());
  }

  GRAPH_DEBUG(
      "(SimplifierUnderContext) after simplify: ",
      std::to_string(simplified_cmp_select_expr));

  analysis::Bound lhs_bound;
  analysis::Bound rhs_bound;
  auto lhs_has_bound = getLoopBoundInfo(simplified_lhs, &lhs_bound);
  auto rhs_has_bound = getLoopBoundInfo(simplified_rhs, &rhs_bound);
  if (!lhs_has_bound || !rhs_has_bound) {
    GRAPH_DEBUG(
        "(SimplifierUnderContext) Final: ",
        std::to_string(simplified_cmp_select_expr));
    return simplified_cmp_select_expr;
  }

  analysis::CmpEvalResult cmp_res =
      analysis::compareBound(lhs_bound, rhs_bound, v->compare_select_op());

  // Return the simplified ret1/ret2 if the compare result is deterministic.
  // Otherwise, return the simplified CompareSelect directly.
  auto ret_expr = (cmp_res == analysis::CmpEvalResult::True)
      ? simplified_ret1
      : ((cmp_res == analysis::CmpEvalResult::False)
             ? simplified_ret2
             : simplified_cmp_select_expr);
  GRAPH_DEBUG("(SimplifierUnderContext) Final: ", std::to_string(ret_expr));
  return ret_expr;
}

ExprPtr SimplifierUnderContext::mutate(ModPtr v) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();

  std::ostringstream oss;
  if (auto ret = distributeMod(lhs, rhs, var_bound_info_)) {
    GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *ret);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    return ret->accept_mutator(this);
  }

  // i % N -> i if the range of i's values is a subset of [0, N)
  // where N is an integer constant
  auto lhsVar = to<Var>(lhs);
  ExprPtr rhsScalar = rhs->isConstant() ? rhs : nullptr;
  if (lhsVar && rhsScalar && !rhsScalar->dtype().is_floating_point()) {
    auto got = var_bound_info_.find(lhsVar);
    if (got != var_bound_info_.end()) {
      auto start = got->second.start;
      auto end = got->second.end;
      ExprPtr check_start = IRSimplifier::simplify(
          alloc<CompareSelect>(start, immLike(start, 0), kGE));
      ExprPtr check_end =
          IRSimplifier::simplify(alloc<CompareSelect>(end, rhsScalar, kLE));
      if (check_start->isConstant() && check_end->isConstant() &&
          immediateEquals(check_start, 1) && immediateEquals(check_end, 1)) {
        GRAPH_DEBUG("SimplifierUnderContext: ", *v, " => ", *lhsVar);
        return lhsVar;
      }
    }
  }

  ExprPtr lhs_new = lhs->accept_mutator(this);
  ExprPtr rhs_new = rhs->accept_mutator(this);
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
  return alloc<Mod>(lhs_new, rhs_new);
}

bool SimplifierUnderContext::getLoopBoundInfo(
    const ExprPtr& expr,
    analysis::Bound* loop_bound_info) {
  if (expr == nullptr)
    return false;

  if (expr->isConstant()) {
    loop_bound_info->start = expr;
    loop_bound_info->end = expr;
    return true;
  }

  VarPtr var_key = to<Var>(expr);
  if (var_key == nullptr) {
    return false;
  }

  auto got = var_bound_info_.find(var_key);
  if (got == var_bound_info_.end()) {
    return false;
  }

  loop_bound_info->start = got->second.start;
  // TODO: Need to add the boundary information(close/open) of a range to
  // Bound. Currently, the VarBoundInfo comes from for-loop statement while
  // the end of the boundary is open. But we assume the start and end of a
  // range are always close. Hence, we explicitly convert the open boundary to
  // close.
  //   [for-start, for-stop) => [for-start, for-stop -1]
  loop_bound_info->end = IRSimplifier::simplify(
      alloc<Sub>(got->second.end, immLike(got->second.end, 1)));
  return true;
}

bool exprEquals(ExprPtr A, ExprPtr B) {
  try {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    ExprPtr diff = IRSimplifier::simplify(alloc<Sub>(A, B));
    if (!diff->isConstant()) {
      return false;
    }
    return immediateEquals(diff, 0);
  } catch (std::exception& e) {
    return false;
  }
}

ExprPtr IRSimplifier::simplify(ExprPtr e) {
  GRAPH_DEBUG("(Simplifier) Original: ", std::to_string(e));
  SimplifierUnderContext ctxsimplifier;
  e = e->accept_mutator(&ctxsimplifier);

  PolynomialTransformer simplifier;
  e = e->accept_mutator(&simplifier);

  // There may be terms left in the IR, expand them.
  TermExpander expander(&simplifier);
  e = e->accept_mutator(&expander);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  if (!expander.check_safe()) {
    throw malformed_input("eliminated null Allocation without free");
  }

  GRAPH_DEBUG("(Simplifier) Simplified: ", std::to_string(e));
  return e;
}

StmtPtr IRSimplifier::simplify(StmtPtr s) {
  GRAPH_DEBUG("(Simplifier) Original: ", std::to_string(s));
  SimplifierUnderContext ctxsimplifier;
  s = s->accept_mutator(&ctxsimplifier);

  PolynomialTransformer simplifier;
  s = s->accept_mutator(&simplifier);
  if (s == nullptr) {
    GRAPH_DEBUG("(Simplifier) Simplified: NULL");
    return nullptr;
  }

  // There may be terms left in the IR, expand them.
  TermExpander expander(&simplifier);
  s = s->accept_mutator(&expander);
  if (!expander.check_safe()) {
    throw malformed_input("eliminated null Allocation without free");
  }

  GRAPH_DEBUG("(Simplifier) Simplified: ", std::to_string(s));
  return s;
}

} // namespace torch::jit::tensorexpr
