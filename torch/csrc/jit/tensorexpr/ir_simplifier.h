#pragma once

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

/* IR Simplification
 *
 * Simplfies expressions in two stages:
 *  1. Recursively traverse the map combining similar operations into Terms
 * (interacted via Multiplication) and Polynomials (interacted via Addition). We
 * reorder the components of each Term or Polynomial into a consistent order to
 * allow combination or cancelling of like terms.
 *  2. Once the format of the tree is minimal, expand each Term into a sequence
 * of Muls, and each Polynomial into a sequence of Ads.
 */

namespace torch {
namespace jit {
namespace tensorexpr {

// A bunch of helpers for determine the Dtype of the output of a multi argument
// Term or Polynomial.
template <class ExprType>
Dtype promoteTypesVec(Expr* s, std::vector<ExprType*>& v) {
  Dtype t = s->dtype();
  bool first = true;

  for (auto* e : v) {
    if (first) {
      t = Dtype(t.scalar_type(), e->dtype().lanes());
      first = false;
    }
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesVec(std::vector<ExprType*>& v) {
  if (v.empty()) {
    throw malformed_input("empty list of types");
  }

  Dtype t = v[0]->dtype();
  for (auto* e : v) {
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesMap(
    Expr* s,
    std::unordered_map<SimplifierHashType, ExprType*>& m) {
  Dtype t = s->dtype();
  bool first = true;
  for (auto& e : m) {
    if (first) {
      t = Dtype(t.scalar_type(), e.second->dtype().lanes());
      first = false;
    }
    t = promoteTypes(t, e.second->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesVar(ExprType* e) {
  return e->dtype();
}

template <class ExprType, class... Args>
Dtype promoteTypesVar(ExprType* e, Args... es) {
  Dtype lhs = e->dtype();
  Dtype rhs = promoteTypesVar(es...);
  if (e->isConstant()) {
    lhs = Dtype(lhs.scalar_type(), rhs.lanes());
  }

  return promoteTypes(lhs, rhs);
}

// Creates a new Expr of the given type with the provided lhs and rhs.
inline Expr* newBinaryOpOfType(
    IRNodeType expr_type,
    Expr* lhs,
    Expr* rhs,
    bool option) {
  switch (expr_type) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case IRNodeType::kAdd:
      return new Add(lhs, rhs);
    case IRNodeType::kSub:
      return new Sub(lhs, rhs);
    case IRNodeType::kMul:
      return new Mul(lhs, rhs);
    case IRNodeType::kDiv:
      return new Div(lhs, rhs);
    case IRNodeType::kMod:
      return new Mod(lhs, rhs);
    case IRNodeType::kMax:
      return new Max(lhs, rhs, option);
    case IRNodeType::kMin:
      return new Min(lhs, rhs, option);
    case IRNodeType::kAnd:
      return new And(lhs, rhs);
    case IRNodeType::kXor:
      return new Xor(lhs, rhs);
    case IRNodeType::kLshift:
      return new Lshift(lhs, rhs);
    case IRNodeType::kRshift:
      return new Rshift(lhs, rhs);
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
      return nullptr;
  }
}

// Uses the evaluator to fold an Expression with constant terms.
// E.g. evaluateOp(Add(3, 4)) => 7.
// Expr v must not have any unbound Vars.
inline Expr* evaluateOp(Expr* v) {
  ExprHandle handle(v);
  ExprEval<SimpleIREvaluator> eval(handle);

  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                 \
  case ScalarType::Name: {                                    \
    Type val = eval.value<Type>();                            \
    return getImmediateByType(v->dtype().scalar_type(), val); \
  }
    AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, TYPE_CASE);
#undef TYPE_CASE
    default:
      LOG(FATAL) << "Unsupported datatype: " << v->dtype();
      return nullptr;
  }
  return nullptr;
}

// A Term represents a grouping of Exprs through multiplication.
// E.g. product(scalar, *variables).
class Term : public ExprNode<Term> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Term(HashProvider& hasher, Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addComponent(ts...);
    sort();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Term(HashProvider& hasher, Expr* s, std::vector<Expr*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor from a map of hash -> var, used when merging Terms.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Term(
      HashProvider& hasher,
      Expr* s,
      std::unordered_map<SimplifierHashType, Expr*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addComponent(p.second);
    }
    sort();
  }

  Expr* scalar() const {
    return scalar_;
  }
  const std::vector<Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  // Produce a hash of just the variable components of this term, to determine
  // if it can be combined with another term.
  SimplifierHashType hashVars() const;

 private:
  std::vector<Expr*> variables_;
  Expr* scalar_;
  HashProvider& hasher_;

  void addComponent() {}
  void addComponent(Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Sort by hash to normalize order of components.
  void sort();
};

// Polynomial represents a grouping of Exprs by addition.
// E.g. sum(*variables, scalar).
// This would better be called Expression, but, naming conflict...
class Polynomial : public ExprNode<Polynomial> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Polynomial(HashProvider& hasher, Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addTerm(ts...);
    sort();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Polynomial(HashProvider& hasher, Expr* s, std::vector<Term*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Helper constructor for list of terms with no scalar component.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Polynomial(HashProvider& hasher, std::vector<Term*> terms)
      : ExprNodeBase(promoteTypesVec(terms)),
        variables_(std::move(terms)),
        scalar_(getImmediateByType(dtype(), 0)),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor for map of hash -> var, used when merging
  // Polynomials.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Polynomial(
      HashProvider& hasher,
      Expr* s,
      std::unordered_map<SimplifierHashType, Term*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addTerm(p.second);
    }
    sort();
  }

  Expr* scalar() const {
    return scalar_;
  }
  const std::vector<Term*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  SimplifierHashType hashVars() const;

 private:
  std::vector<Term*> variables_;
  Expr* scalar_;
  HashProvider& hasher_;

  void addTerm(Term* t) {
    variables_.push_back(t);
  }
  template <class... Ts>
  void addTerm(Term* t, Ts... ts) {
    addTerm(t);
    addTerm(ts...);
  }

  // Sort by hash to normalize order of terms.
  void sort();
};

class RoundOff : public BinaryOpNode<RoundOff> {
 public:
  RoundOff(Expr* lhs, Expr* rhs) : BinaryOpNode(lhs, rhs, IRNodeType::kOther) {}
};

class MaxTerm : public ExprNode<MaxTerm> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MaxTerm(HashProvider& hasher, Expr* s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MaxTerm(HashProvider& hasher, Expr* s, bool p, std::vector<Expr*> v)
      : ExprNodeBase(s ? promoteTypesVec(s, v) : promoteTypesVec(v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    uniquefy();
  }

  bool propagate_nans() const {
    return propagate_nans_;
  }

  Expr* scalar() const {
    return scalar_;
  }
  const std::vector<Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<Expr*> variables_;
  Expr* scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

class MinTerm : public ExprNode<MinTerm> {
 public:
  template <class... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MinTerm(HashProvider& hasher, Expr* s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  MinTerm(HashProvider& hasher, Expr* s, bool p, std::vector<Expr*> v)
      : ExprNodeBase(s ? promoteTypesVec(s, v) : promoteTypesVec(v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    uniquefy();
  }

  bool propagate_nans() const {
    return propagate_nans_;
  }

  Expr* scalar() const {
    return scalar_;
  }
  const std::vector<Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<Expr*> variables_;
  Expr* scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

// Context-sensitive IR simplification
using VarBoundInfo = std::unordered_map<Var*, std::pair<Expr*, Expr*>>;
class TORCH_API SimplifierUnderContext : public IRMutator {
 public:
  ~SimplifierUnderContext() override = default;
  // Add boundary info for index variables in for-loops
  Stmt* mutate(For* v) override;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_;
  VarBoundInfo var_bound_info_;
};

// Stmt simplification should occur in both modes.
class TORCH_API PolynomialBase : public IRMutator {
 public:
  ~PolynomialBase() override = default;

  Stmt* mutate(Block* v) override;

  Stmt* mutate(Cond* v) override;

  Stmt* mutate(For* v) override;

  // Trivially factorize terms by GCD of scalar components.
  Term* factorizePolynomial(Polynomial* poly);

  HashProvider& hasher() {
    return hasher_;
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_;
};

// Simplify the IR by combining arithmetic expressions over common terms.
class TORCH_API PolynomialTransformer : public PolynomialBase {
 public:
  using PolynomialBase::mutate;
  // Inserts term into the provided map, in the case of a hash collision
  // combines the term with the existing and updates the map.
  void addOrUpdateTerm(
      std::unordered_map<SimplifierHashType, Term*>& varmap,
      Term* term);

  // Add Polynomial expressions, combining Terms representing the same
  // variables.
  Expr* addPolynomials(Polynomial* lhs, Polynomial* rhs);

  // Insert a new Term into the provided polynomial. If the new term has common
  // variables to an existing term it is combined.
  Expr* insertTerm(Polynomial* poly, Term* term);

  // Merge and simplify addition.
  Expr* mutate(Add* v) override;

  // Subtract one term from another, cancelling if necessary.
  Expr* subTerms(Term* lhs, Term* rhs, bool negated);

  // Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
  // possible.
  Expr* subPolynomials(Polynomial* lhs, Polynomial* rhs);

  // Merge and simplify subtraction.
  Expr* mutate(Sub* v) override;

  // Multiply two terms together, usually creating a new term with the variable
  // lists concatenated.
  Term* mulTerms(Term* lhs, Term* rhs);

  // Multiply a Polynomial by a Term.
  Expr* polyByTerm(Polynomial* poly, Term* term);

  // Match a rounding pattern and create a RoundOff if found.
  Expr* isRoundOff(Expr* lhs, Expr* rhs);

  // Inserts a new component into a term, simplifying if possible.
  Expr* insertIntoTerm(Term* term, Expr* expr);

  // Merge and simplify multiplication.
  Expr* mutate(Mul* v) override;

  Expr* mutate(Div* v) override;

  Expr* mutate(Mod* v) override;

  Expr* mutate(And* v) override {
    return mutateBinaryOp(v, this);
  }

  Expr* mutate(Xor* v) override {
    return mutateBinaryOp(v, this);
  }

  Expr* mutate(Lshift* v) override {
    return mutateBinaryOp(v, this);
  }

  Expr* mutate(Rshift* v) override {
    return mutateBinaryOp(v, this);
  }

  Expr* mutate(Max* v) override;

  Expr* mutate(Min* v) override;

  Expr* mutate(CompareSelect* v) override;

  Expr* mutate(Intrinsics* v) override;

  Expr* mutate(Cast* v) override;

  Expr* mutate(IfThenElse* v) override;

  template <typename Op>
  static Expr* mutateBinaryOp(
      BinaryOpNode<Op>* v,
      IRMutator* mutator,
      bool option = false) {
    Expr* lhs = v->lhs();
    Expr* rhs = v->rhs();
    Expr* lhs_new = lhs->accept_mutator(mutator);
    Expr* rhs_new = rhs->accept_mutator(mutator);

    Expr* node = v;

    if (lhs != lhs_new || rhs != rhs_new) {
      node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
    }

    // Can only fold if both sides are constant.
    if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
      return node;
    }

    return evaluateOp(node);
  }

  static Expr* simplify(Expr* e);
  static ExprHandle simplify(const ExprHandle& e);
  static Stmt* simplify(Stmt* e);
};

// Expands Terms and Polynomial expressions into primitive operations.
// Does some simple factorization and reordering.
class TORCH_API TermExpander : public PolynomialBase {
  PolynomialTransformer* simplifier_;
  std::set<Var*> eliminated_allocations_;

 public:
  using PolynomialBase::mutate;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TermExpander(PolynomialTransformer* simplifier) : simplifier_(simplifier) {}
  bool check_safe() {
    return eliminated_allocations_.empty();
  }

  // Expand Terms out to a series of Muls.
  Expr* mutate(Term* v) override;

  // Expand Polynomials out to a series of Adds.
  Expr* mutate(Polynomial* v) override;

  // Expand MaxTerms to a series of Max ops.
  Expr* mutate(MaxTerm* v) override;

  // Expand MinTerms to a series of Min ops.
  Expr* mutate(MinTerm* v) override;

  // Expand RoundOff to it's component: Mul(Div(lhs, rhs), rhs).
  Expr* mutate(RoundOff* v) override;

  // Eliminate zero length allocations.
  Stmt* mutate(Allocate* v) override;
  Stmt* mutate(Free* v) override;

  // Override to enable condition fusing.
  Block* fuseConditions(Block* v);
  Stmt* fuseSyncThreads(Block* block);
  Stmt* mutate(Block* v) override;
};

class TORCH_API IRSimplifier {
 public:
  static Expr* simplify(Expr* e) {
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

    return e;
  }

  static ExprHandle simplify(const ExprHandle& e) {
    return ExprHandle(simplify(e.node()));
  }

  static Stmt* simplify(Stmt* s) {
    SimplifierUnderContext ctxsimplifier;
    s = s->accept_mutator(&ctxsimplifier);

    PolynomialTransformer simplifier;
    s = s->accept_mutator(&simplifier);
    if (s == nullptr) {
      return nullptr;
    }

    // There may be terms left in the IR, expand them.
    TermExpander expander(&simplifier);
    s = s->accept_mutator(&expander);
    if (!expander.check_safe()) {
      throw malformed_input("eliminated null Allocation without free");
    }

    return s;
  }
};

// Flattens the buf and performs the simplifier on the flattened dims.
Expr* buf_flat_size(Buf* v);
// Returns true if expressions A and B can be simplified to an equal expression.
TORCH_API bool exprEquals(Expr* A, Expr* B);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
