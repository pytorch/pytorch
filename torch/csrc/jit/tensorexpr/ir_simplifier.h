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
Dtype promoteTypesVec(const Expr* s, std::vector<const ExprType*>& v) {
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
Dtype promoteTypesVec(std::vector<const ExprType*>& v) {
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
    const Expr* s,
    std::unordered_map<SimplifierHashType, const ExprType*>& m) {
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
Dtype promoteTypesVar(const ExprType* e) {
  return e->dtype();
}

template <class ExprType, class... Args>
Dtype promoteTypesVar(const ExprType* e, Args... es) {
  Dtype lhs = e->dtype();
  Dtype rhs = promoteTypesVar(es...);
  if (e->isConstant()) {
    lhs = Dtype(lhs.scalar_type(), rhs.lanes());
  }

  return promoteTypes(lhs, rhs);
}

// Creates a new Expr of the given type with the provided lhs and rhs.
inline const Expr* newBinaryOpOfType(
    IRNodeType expr_type,
    const Expr* lhs,
    const Expr* rhs,
    bool option) {
  switch (expr_type) {
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
inline Expr* evaluateOp(const Expr* v) {
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
  Term(HashProvider& hasher, const Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addComponent(ts...);
    sort();
  }

  Term(HashProvider& hasher, const Expr* s, std::vector<const Expr*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor from a map of hash -> var, used when merging Terms.
  Term(
      HashProvider& hasher,
      const Expr* s,
      std::unordered_map<SimplifierHashType, const Expr*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addComponent(p.second);
    }
    sort();
  }

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  // Produce a hash of just the variable components of this term, to determine
  // if it can be combined with another term.
  SimplifierHashType hashVars() const;

 private:
  std::vector<const Expr*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;

  void addComponent() {}
  void addComponent(const Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(const Expr* e, Es... es) {
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
  Polynomial(HashProvider& hasher, const Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addTerm(ts...);
    sort();
  }

  Polynomial(HashProvider& hasher, const Expr* s, std::vector<const Term*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Helper constructor for list of terms with no scalar component.
  Polynomial(HashProvider& hasher, std::vector<const Term*> terms)
      : ExprNodeBase(promoteTypesVec(terms)),
        variables_(std::move(terms)),
        scalar_(getImmediateByType(dtype(), 0)),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor for map of hash -> var, used when merging
  // Polynomials.
  Polynomial(
      HashProvider& hasher,
      const Expr* s,
      std::unordered_map<SimplifierHashType, const Term*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addTerm(p.second);
    }
    sort();
  }

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Term*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  SimplifierHashType hashVars() const;

 private:
  std::vector<const Term*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;

  void addTerm(const Term* t) {
    variables_.push_back(t);
  }
  template <class... Ts>
  void addTerm(const Term* t, Ts... ts) {
    addTerm(t);
    addTerm(ts...);
  }

  // Sort by hash to normalize order of terms.
  void sort();
};

class RoundOff : public BinaryOpNode<RoundOff> {
 public:
  RoundOff(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kOther) {}
};

class MaxTerm : public ExprNode<MaxTerm> {
 public:
  template <class... Args>
  MaxTerm(HashProvider& hasher, const Expr* s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  MaxTerm(
      HashProvider& hasher,
      const Expr* s,
      bool p,
      std::vector<const Expr*> v)
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

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<const Expr*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(const Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(const Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

class MinTerm : public ExprNode<MinTerm> {
 public:
  template <class... Args>
  MinTerm(HashProvider& hasher, const Expr* s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  MinTerm(
      HashProvider& hasher,
      const Expr* s,
      bool p,
      std::vector<const Expr*> v)
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

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<const Expr*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(const Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(const Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

// Stmt simplification should occur in both modes.
class TORCH_API IRSimplifierBase : public IRMutator {
 public:
  ~IRSimplifierBase() override = default;

  Stmt* mutate(const Block* v) override;

  Stmt* mutate(const Cond* v) override;

  Stmt* mutate(const For* v) override;

  // Trivially factorize terms by GCD of scalar components.
  const Term* factorizePolynomial(const Polynomial* poly);

  HashProvider& hasher() {
    return hasher_;
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_;
};

// Simplify the IR by combining arithmetic expressions over common terms.
class TORCH_API PolynomialTransformer : public IRSimplifierBase {
 public:
  using IRSimplifierBase::mutate;
  // Inserts term into the provided map, in the case of a hash collision
  // combines the term with the existing and updates the map.
  void addOrUpdateTerm(
      std::unordered_map<SimplifierHashType, const Term*>& varmap,
      const Term* term);

  // Add Polynomial expressions, combining Terms representing the same
  // variables.
  const Expr* addPolynomials(const Polynomial* lhs, const Polynomial* rhs);

  // Insert a new Term into the provided polynomial. If the new term has common
  // variables to an existing term it is combined.
  const Expr* insertTerm(const Polynomial* poly, const Term* term);

  // Merge and simplify addition.
  const Expr* mutate(const Add* v) override;

  // Subtract one term from another, cancelling if necessary.
  const Expr* subTerms(const Term* lhs, const Term* rhs, bool negated);

  // Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
  // possible.
  const Expr* subPolynomials(const Polynomial* lhs, const Polynomial* rhs);

  // Merge and simplify subtraction.
  const Expr* mutate(const Sub* v) override;

  // Multiply two terms together, usually creating a new term with the variable
  // lists concatenated.
  const Term* mulTerms(const Term* lhs, const Term* rhs);

  // Multiply a Polynomial by a Term.
  const Expr* polyByTerm(const Polynomial* poly, const Term* term);

  // Match a rounding pattern and create a RoundOff if found.
  const Expr* isRoundOff(const Expr* lhs, const Expr* rhs);

  // Inserts a new component into a term, simplifying if possible.
  const Expr* insertIntoTerm(const Term* term, const Expr* expr);

  // Merge and simplify multiplication.
  const Expr* mutate(const Mul* v) override;

  const Expr* mutate(const Div* v) override;

  const Expr* mutate(const Mod* v) override;

  const Expr* mutate(const And* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Xor* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Lshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Rshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Max* v) override;

  const Expr* mutate(const Min* v) override;

  const Expr* mutate(const CompareSelect* v) override;

  const Expr* mutate(const Intrinsics* v) override;

  const Expr* mutate(const Cast* v) override;

  const Expr* mutate(const IfThenElse* v) override;

  template <typename Op>
  static const Expr* mutateBinaryOp(
      const BinaryOpNode<Op>* v,
      IRMutator* mutator,
      bool option = false) {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(mutator);
    const Expr* rhs_new = rhs->accept_mutator(mutator);

    const Expr* node = v;

    if (lhs != lhs_new || rhs != rhs_new) {
      node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
    }

    // Can only fold if both sides are constant.
    if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
      return node;
    }

    return evaluateOp(node);
  }

  static const Expr* simplify(const Expr* e);
  static ExprHandle simplify(const ExprHandle& e);
  static Stmt* simplify(Stmt* e);
};

// Expands Terms and Polynomial expressions into primitive operations.
// Does some simple factorization and reordering.
class TORCH_API TermExpander : public IRSimplifierBase {
  PolynomialTransformer* simplifier_;
  std::set<const Var*> eliminated_allocations_;

 public:
  using IRSimplifierBase::mutate;
  TermExpander(PolynomialTransformer* simplifier) : simplifier_(simplifier) {}
  bool check_safe() {
    return eliminated_allocations_.empty();
  }

  // Expand Terms out to a series of Muls.
  const Expr* mutate(const Term* v) override;

  // Expand Polynomials out to a series of Adds.
  const Expr* mutate(const Polynomial* v) override;

  // Expand MaxTerms to a series of Max ops.
  const Expr* mutate(const MaxTerm* v) override;

  // Expand MinTerms to a series of Min ops.
  const Expr* mutate(const MinTerm* v) override;

  // Expand RoundOff to it's component: Mul(Div(lhs, rhs), rhs).
  const Expr* mutate(const RoundOff* v) override;

  // Eliminate zero length allocations.
  Stmt* mutate(const Allocate* v) override;
  Stmt* mutate(const Free* v) override;

  // Override to enable condition fusing.
  Block* fuseConditions(Block* v);
  Stmt* fuseSyncThreads(Block* block);
  Stmt* mutate(const Block* v) override;
};

class TORCH_API IRSimplifier {
 public:
  static const Expr* simplify(const Expr* e) {
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
const Expr* buf_flat_size(const Buf* v);
// Returns true if expressions A and B can be simplified to an equal expression.
TORCH_API bool exprEquals(const Expr* A, const Expr* B);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
