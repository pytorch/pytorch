#pragma once

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <utility>

/* IR Simplification
 *
 * Simplifies expressions in two stages:
 *  1. Recursively traverse the map combining similar operations into Terms
 * (interacted via Multiplication) and Polynomials (interacted via Addition). We
 * reorder the components of each Term or Polynomial into a consistent order to
 * allow combination or cancelling of like terms.
 *  2. Once the format of the tree is minimal, expand each Term into a sequence
 * of Muls, and each Polynomial into a sequence of Ads.
 */

namespace torch::jit::tensorexpr {

// A bunch of helpers for determine the Dtype of the output of a multi argument
// Term or Polynomial.
template <class ExprType>
Dtype promoteTypesVec(const ExprPtr& s, const std::vector<ExprType>& v) {
  Dtype t = s->dtype();
  bool first = true;

  for (const auto& e : v) {
    if (first) {
      t = Dtype(t.scalar_type(), e->dtype().lanes());
      first = false;
    }
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesVec(const std::vector<ExprType>& v) {
  if (v.empty()) {
    throw malformed_input("empty list of types");
  }

  Dtype t = v[0]->dtype();
  for (const auto& e : v) {
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesMap(
    const ExprPtr& s,
    std::unordered_map<SimplifierHashType, ExprType>& m) {
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
Dtype promoteTypesVar(ExprType e) {
  return e->dtype();
}

template <class ExprType, class... Args>
Dtype promoteTypesVar(ExprType e, Args... es) {
  Dtype lhs = e->dtype();
  Dtype rhs = promoteTypesVar(es...);
  if (e->isConstant()) {
    lhs = Dtype(lhs.scalar_type(), rhs.lanes());
  }

  return promoteTypes(lhs, rhs);
}

// Uses the evaluator to fold an Expression with constant terms.
// E.g. evaluateOp(Add(3, 4)) => 7.
// Expr v must not have any unbound Vars.
inline ExprPtr evaluateOp(const ExprPtr& v) {
  ExprHandle handle(v);
  ExprEval<SimpleIREvaluator> eval(handle);

  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                 \
  case ScalarType::Name: {                                    \
    Type val = eval.value<Type>();                            \
    return getImmediateByType(v->dtype().scalar_type(), val); \
  }
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE)
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
  Term(HashProvider& hasher, ExprPtr s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addComponent(ts...);
    sort();
  }

  Term(HashProvider& hasher, ExprPtr s, std::vector<ExprPtr> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(std::move(s)),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor from a map of hash -> var, used when merging Terms.
  Term(
      HashProvider& hasher,
      const ExprPtr& s,
      std::unordered_map<SimplifierHashType, ExprPtr> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addComponent(p.second);
    }
    sort();
  }

  ExprPtr scalar() const {
    return scalar_;
  }
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  // Produce a hash of just the variable components of this term, to determine
  // if it can be combined with another term.
  SimplifierHashType hashVars() const;

 private:
  std::vector<ExprPtr> variables_;
  ExprPtr scalar_;
  HashProvider& hasher_;

  void addComponent() {}
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e));
    addComponent(std::forward<Es>(es)...);
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
  Polynomial(HashProvider& hasher, ExprPtr s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addTerm(ts...);
    sort();
  }

  Polynomial(HashProvider& hasher, const ExprPtr& s, std::vector<TermPtr> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Helper constructor for list of terms with no scalar component.
  Polynomial(HashProvider& hasher, std::vector<TermPtr> terms)
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
      const ExprPtr& s,
      std::unordered_map<SimplifierHashType, TermPtr> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addTerm(p.second);
    }
    sort();
  }

  ExprPtr scalar() const {
    return scalar_;
  }
  const std::vector<TermPtr>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  SimplifierHashType hashVars() const;

 private:
  std::vector<TermPtr> variables_;
  ExprPtr scalar_;
  HashProvider& hasher_;

  void addTerm(TermPtr t) {
    variables_.push_back(std::move(t));
  }
  template <class... Ts>
  void addTerm(TermPtr t, Ts&&... ts) {
    addTerm(std::move(t));
    addTerm(std::forward<Ts>(ts)...);
  }

  // Sort by hash to normalize order of terms.
  void sort();
};

class RoundOff : public BinaryOpNode<RoundOff> {
 public:
  RoundOff(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kOther) {}
};

class MaxTerm : public ExprNode<MaxTerm> {
 public:
  template <class... Args>
  MaxTerm(HashProvider& hasher, ExprPtr s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  MaxTerm(
      HashProvider& hasher,
      const ExprPtr& s,
      bool p,
      std::vector<ExprPtr> v)
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

  ExprPtr scalar() const {
    return scalar_;
  }
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<ExprPtr> variables_;
  ExprPtr scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e));
    addComponent(std::forward<Es>(es)...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

class MinTerm : public ExprNode<MinTerm> {
 public:
  template <class... Args>
  MinTerm(HashProvider& hasher, ExprPtr s, bool p, Args... ts)
      : ExprNodeBase(s ? promoteTypesVar(s, ts...) : promoteTypesVar(ts...)),
        scalar_(s),
        hasher_(hasher),
        propagate_nans_(p) {
    addComponent(ts...);
    uniquefy();
  }

  MinTerm(
      HashProvider& hasher,
      const ExprPtr& s,
      bool p,
      std::vector<ExprPtr> v)
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

  ExprPtr scalar() const {
    return scalar_;
  }
  const std::vector<ExprPtr>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

 private:
  std::vector<ExprPtr> variables_;
  ExprPtr scalar_;
  HashProvider& hasher_;
  bool propagate_nans_;

  void addComponent() {}
  void addComponent(ExprPtr e) {
    variables_.push_back(std::move(e));
  }
  template <class... Es>
  void addComponent(ExprPtr e, Es&&... es) {
    addComponent(std::move(e));
    addComponent(std::forward<Es>(es)...);
  }

  // Uniquefy the terms using their hash.
  void uniquefy();
};

// Context-sensitive IR simplification
using VarBoundInfo = std::unordered_map<VarPtr, analysis::Bound>;

class TORCH_API SimplifierUnderContext : public IRMutator {
 public:
  ~SimplifierUnderContext() override = default;
  // Add boundary info for index variables in for-loops
  StmtPtr mutate(const ForPtr& v) override;

  ExprPtr mutate(const DivPtr& v) override;
  ExprPtr mutate(const ModPtr& v) override;
  ExprPtr mutate(const CompareSelectPtr& v) override;
  ExprPtr mutate(const IfThenElsePtr& v) override;

 protected:
  bool getLoopBoundInfo(const ExprPtr& expr, analysis::Bound* loop_bound_info);

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  HashProvider hasher_;
  VarBoundInfo var_bound_info_;
};

// Stmt simplification should occur in both modes.
class TORCH_API PolynomialBase : public IRMutator {
 public:
  ~PolynomialBase() override = default;

  StmtPtr mutate(const BlockPtr& v) override;

  StmtPtr mutate(const CondPtr& v) override;

  StmtPtr mutate(const ForPtr& v) override;

  // Trivially factorize terms by GCD of scalar components.
  TermPtr factorizePolynomial(const PolynomialPtr& poly);

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
      std::unordered_map<SimplifierHashType, TermPtr>& varmap,
      const TermPtr& term);

  // Add Polynomial expressions, combining Terms representing the same
  // variables.
  ExprPtr addPolynomials(const PolynomialPtr& lhs, const PolynomialPtr& rhs);

  // Insert a new Term into the provided polynomial. If the new term has
  // common variables to an existing term it is combined.
  ExprPtr insertTerm(const PolynomialPtr& poly, const TermPtr& term);

  // Merge and simplify addition.
  ExprPtr mutate(const AddPtr& v) override;

  // Subtract one term from another, cancelling if necessary.
  ExprPtr subTerms(const TermPtr& lhs, TermPtr rhs, bool negated);

  // Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
  // possible.
  ExprPtr subPolynomials(const PolynomialPtr& lhs, const PolynomialPtr& rhs);

  // Merge and simplify subtraction.
  ExprPtr mutate(const SubPtr& v) override;

  // Multiply two terms together, usually creating a new term with the variable
  // lists concatenated.
  TermPtr mulTerms(const TermPtr& lhs, const TermPtr& rhs);

  // Multiply a Polynomial by a Term.
  ExprPtr polyByTerm(const PolynomialPtr& poly, const TermPtr& term);

  // Match a rounding pattern and create a RoundOff if found.
  ExprPtr isRoundOff(const ExprPtr& lhs, const ExprPtr& rhs);

  // Inserts a new component into a term, simplifying if possible.
  ExprPtr insertIntoTerm(const TermPtr& term, const ExprPtr& expr);

  // Merge and simplify multiplication.
  ExprPtr mutate(const MulPtr& v) override;

  ExprPtr mutate(const DivPtr& v) override;

  ExprPtr mutate(const ModPtr& v) override;

  ExprPtr mutate(const AndPtr& v) override;

  ExprPtr mutate(const XorPtr& v) override;

  ExprPtr mutate(const LshiftPtr& v) override;

  ExprPtr mutate(const RshiftPtr& v) override;

  ExprPtr mutate(const MaxPtr& v) override;

  ExprPtr mutate(const MinPtr& v) override;

  ExprPtr mutate(const CompareSelectPtr& v) override;

  ExprPtr mutate(const IntrinsicsPtr& v) override;

  ExprPtr mutate(const CastPtr& v) override;

  ExprPtr mutate(const IfThenElsePtr& v) override;

  static ExprPtr simplify(ExprPtr e);
  static ExprHandle simplify(const ExprHandle& e);
  static StmtPtr simplify(StmtPtr e);
};

// Expands Terms and Polynomial expressions into primitive operations.
// Does some simple factorization and reordering.
class TORCH_API TermExpander : public PolynomialBase {
  PolynomialTransformer* simplifier_;
  std::set<VarPtr> eliminated_allocations_;

 public:
  using PolynomialBase::mutate;
  TermExpander(PolynomialTransformer* simplifier) : simplifier_(simplifier) {}
  bool check_safe() {
    return eliminated_allocations_.empty();
  }

  // Expand Terms out to a series of Muls.
  ExprPtr mutate(const TermPtr& v) override;

  // Expand Polynomials out to a series of Adds.
  ExprPtr mutate(const PolynomialPtr& v) override;

  // Expand MaxTerms to a series of Max ops.
  ExprPtr mutate(const MaxTermPtr& v) override;

  // Expand MinTerms to a series of Min ops.
  ExprPtr mutate(const MinTermPtr& v) override;

  // Expand RoundOff to it's component: Mul(Div(lhs, rhs), rhs).
  ExprPtr mutate(const RoundOffPtr& v) override;

  // Eliminate zero length allocations.
  StmtPtr mutate(const AllocatePtr& v) override;
  StmtPtr mutate(const FreePtr& v) override;

  // Override to enable condition fusing.
  BlockPtr fuseConditions(BlockPtr v);
  StmtPtr fuseSyncThreads(BlockPtr block);
  StmtPtr mutate(const BlockPtr& v) override;
};

class TORCH_API IRSimplifier {
 public:
  static StmtPtr simplify(StmtPtr s);
  static ExprPtr simplify(ExprPtr e);
  static ExprHandle simplify(const ExprHandle& e) {
    return ExprHandle(simplify(e.node()));
  }
};

// Flattens the buf and performs the simplifier on the flattened dims.
ExprPtr buf_flat_size(const BufPtr& v);
// Returns true if expressions A and B can be simplified to an equal expression.
TORCH_API bool exprEquals(const ExprPtr& A, const ExprPtr& B);

} // namespace torch::jit::tensorexpr
