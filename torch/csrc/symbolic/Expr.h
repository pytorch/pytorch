#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Native port of the subset of sympy used by torch's symbolic shapes. Every
// construction rule is a port of the corresponding sympy (1.14) code, so a
// native expression converts to a sympy expression that is == to what Python
// builds. Anything outside the ported subset throws NativeUnsupported and the
// caller falls back to Python.

namespace torch::symbolic {

struct NativeUnsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Kind : uint8_t {
  Integer,
  Rational,
  IntInfinity,
  NegativeIntInfinity,
  Symbol,
  Pow,
  Mul,
  Add,
  // Custom functions from torch/utils/_sympy/functions.py (Functions.cpp).
  Mod,
  PythonMod,
  FloorDiv,
  CleanDiv,
  // Boolean kinds: sympy Booleans that are not Exprs.
  BooleanTrue,
  BooleanFalse,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  Not,
  And,
  Or,
};

enum class Tri : int8_t { False = 0, True = 1, Unknown = 2 };

enum class Fact : uint8_t {
  commutative,
  integer,
  noninteger,
  rational,
  irrational,
  real,
  extended_real,
  finite,
  infinite,
  zero,
  nonzero,
  positive,
  negative,
  nonnegative,
  nonpositive,
  extended_positive,
  extended_negative,
  extended_nonnegative,
  extended_nonpositive,
  extended_nonzero,
  even,
  odd,
  prime,
  composite,
  algebraic,
  transcendental,
  complex,
  imaginary,
  hermitian,
  antihermitian,
  polar,
  NumFacts,
};

constexpr size_t kNumFacts = static_cast<size_t>(Fact::NumFacts);
using Facts = std::array<Tri, kNumFacts>;

const char* fact_name(Fact f);

// A sympy FactKB (obj._assumptions) as bitmasks over Fact. A fact is in the KB
// when its bit is in `known`; it may still be neither true nor false (sympy
// stores None for a fact it asked about and could not decide).
struct FactKB {
  uint32_t true_mask = 0;
  uint32_t false_mask = 0;
  uint32_t known = 0;

  Tri get(Fact f) const {
    uint32_t b = 1u << static_cast<unsigned>(f);
    return (true_mask & b) ? Tri::True : (false_mask & b) ? Tri::False : Tri::Unknown;
  }
};

// FactKB.deduce_all_facts.
void deduce_all_facts(FactKB& kb, c10::ArrayRef<std::pair<Fact, bool>> facts);

// sympy.core.assumptions._assume_rules as bitmasks over Fact.
struct Implication {
  uint32_t true_mask;
  uint32_t false_mask;
};
struct BetaRule {
  uint32_t cond_true;
  uint32_t cond_false;
  Fact fact;
  bool value;
};
struct AssumeRules {
  // Indexed by [fact][value].
  c10::ArrayRef<std::array<Implication, 2>> full_implications;
  c10::ArrayRef<BetaRule> beta_rules;
  c10::ArrayRef<std::array<uint64_t, 2>> beta_triggers;
  c10::ArrayRef<uint32_t> prereq;
};
AssumeRules assume_rules();

struct Expr {
  Kind kind;
  uint32_t id;
  size_t hash;
  // Integer: value in p. Rational: p/q in lowest terms, q > 1. Symbol: index
  // into ExprArena's symbol table in p.
  int64_t p;
  int64_t q;
  // Add/Mul: sympy's order (coefficient first, the rest by Basic.compare).
  // Pow: base, exponent. Relationals: lhs, rhs. And/Or: sympy's ordered().
  c10::SmallVector<const Expr*, 3> args;
  // Assumptions cache, like sympy's obj._assumptions.
  mutable FactKB kb;

  bool is_number() const {
    return kind <= Kind::NegativeIntInfinity;
  }
  bool is_rational() const {
    return kind == Kind::Integer || kind == Kind::Rational;
  }
  // isinstance(e, Boolean) for everything but Symbol, which is both an Expr
  // and a Boolean in sympy.
  bool is_boolean() const {
    return kind >= Kind::BooleanTrue;
  }
  bool is_relational() const {
    return kind >= Kind::Eq && kind <= Kind::Ge;
  }
  bool is_function() const {
    return kind > Kind::Add && kind < Kind::BooleanTrue;
  }
};

// The class name of a function kind.
const char* function_name(Kind k);

// A sympy sort key (Basic.sort_key, default_sort_key): nested tuples of ints,
// strings and Numbers, compared like Python tuples. Subkeys are shared, as the
// tuples of sympy's cached sort_key are.
struct SortKey;
using SortKeyPtr = std::shared_ptr<const SortKey>;
struct SortKey {
  enum class Type : uint8_t { Int, Str, Num, Tuple };
  Type type;
  int64_t i = 0;
  std::string s;
  const Expr* num = nullptr;
  std::vector<SortKeyPtr> items;
};

// Python's three-way tuple comparison; unorderable types throw.
int compare_keys(const SortKey& a, const SortKey& b);

struct Num {
  int64_t p;
  int64_t q;
};

struct SymbolInfo {
  std::string name;
  // The deduced assumptions (sympy's assumptions0).
  FactKB facts;
  // sympy's Dummy.dummy_index; 0 for a Symbol.
  uint64_t dummy_index = 0;
};

class ExprArena : public c10::intrusive_ptr_target {
 public:
  ExprArena();

  const Expr* integer(int64_t v);
  const Expr* rational(int64_t p, int64_t q);
  const Expr* int_oo() const {
    return int_oo_;
  }
  const Expr* neg_int_oo() const {
    return neg_int_oo_;
  }
  const Expr* boolean(bool v) const {
    return v ? true_ : false_;
  }
  const Expr* symbol(const std::string& name, const Facts& facts);
  // A fresh Dummy(name, **{fact: True}).
  const Expr* dummy(const std::string& name, Fact fact);
  const SymbolInfo& symbol_info(const Expr* e) const {
    return symbols_.at(e->p);
  }

  // Add(*args), Mul(*args), Pow(b, e), -a and a - b with sympy's evaluation.
  const Expr* add(c10::ArrayRef<const Expr*> args);
  const Expr* mul(c10::ArrayRef<const Expr*> args);
  const Expr* pow(const Expr* b, const Expr* e);
  const Expr* neg(const Expr* a);
  const Expr* sub(const Expr* a, const Expr* b);
  // cls(*args) for a function kind: cls.eval's result, or the unevaluated
  // node. Nodes whose args are all numbers throw, so a node without free
  // symbols is always a Number.
  const Expr* function(Kind kind, c10::ArrayRef<const Expr*> args);

  // Eq/Ne/Lt/Le/Gt/Ge(lhs, rhs, evaluate=evaluate) (Relational.cpp). These are
  // the sympy constructors, not IntInfinity's __ge__ etc. operator overloads.
  const Expr* rel(Kind kind, const Expr* lhs, const Expr* rhs, bool evaluate = true);
  // Not(a), And(*args) and Or(*args).
  const Expr* logical_not(const Expr* a);
  const Expr* logical_and(c10::ArrayRef<const Expr*> args);
  const Expr* logical_or(c10::ArrayRef<const Expr*> args);
  // The And/Or with exactly these args, which must be those of an And/Or that
  // sympy built. Not a constructor: Or's filter is not idempotent.
  const Expr* lattice_from_args(Kind kind, c10::ArrayRef<const Expr*> args);
  // The Relational properties of the same names.
  const Expr* reversed(const Expr* r);
  const Expr* reversedsign(const Expr* r);
  const Expr* negated(const Expr* r);
  const Expr* weak(const Expr* r);
  const Expr* strict(const Expr* r);
  // sympy.core.relational.is_eq and is_ge.
  Tri is_eq(const Expr* lhs, const Expr* rhs);
  Tri is_ge(const Expr* lhs, const Expr* rhs);

  // expr.is_<fact>, ported from sympy's _ask and the _eval_is_* handlers.
  Tri ask(const Expr* e, Fact f);
  // Basic.compare.
  int compare(const Expr* a, const Expr* b) const;

  // e.sort_key() and sympy.core.sorting.ordered(seq) with the default keys
  // (Sorting.cpp).
  const SortKeyPtr& sort_key(const Expr* e);
  std::vector<const Expr*> ordered(c10::ArrayRef<const Expr*> seq);
  c10::SmallVector<const Expr*, 4> as_ordered_terms(const Expr* e);
  c10::SmallVector<const Expr*, 4> as_ordered_factors(const Expr* e);
  bool could_extract_minus_sign(const Expr* e);
  // Relational.canonical.
  const Expr* canonical(const Expr* r);

  // str(e): sympy's StrPrinter (Printer.cpp).
  std::string str(const Expr* e);

  // Ports of the sympy helpers behind the Add sign handlers (ExprTools.cpp).
  std::pair<const Expr*, const Expr*> as_coeff_Add(const Expr* e);
  c10::SmallVector<const Expr*, 4> free_symbols(const Expr* e) const;
  bool is_polynomial(const Expr* e) const;
  const Expr* diff(const Expr* e, const Expr* x);
  const Expr* xreplace(const Expr* e, c10::ArrayRef<std::pair<const Expr*, const Expr*>> reps);
  std::pair<const Expr*, const Expr*> as_numer_denom(const Expr* e);
  // sympy.core.exprtools._monotonic_sign; nullptr for None.
  const Expr* monotonic_sign(const Expr* e);

  size_t size() const {
    return storage_.size();
  }

 private:
  friend class StrPrinter;

  const Expr* number(Num n);
  const Expr* intern(Kind kind, int64_t p, int64_t q, c10::ArrayRef<const Expr*> args);
  // Assoc node from already-processed args, like AssocOp._from_args.
  const Expr* from_args(Kind kind, c10::SmallVectorImpl<const Expr*>& args);
  const Expr* number_pow(Num b, int64_t e);
  // Mod.eval and PythonMod.eval; nullptr for None.
  const Expr* eval_mod(Kind kind, const Expr* p, const Expr* q);
  // FloorDiv.eval, also for CleanDiv; nullptr for None.
  const Expr* eval_floordiv(const Expr* base, const Expr* divisor);
  const Expr* new_symbol(const std::string& name, const FactKB& kb);
  const Expr* as_boolean(const Expr* e);
  // The tail of LatticeOp.__new__.
  const Expr* lattice(Kind kind, c10::ArrayRef<const Expr*> args);
  const Expr* keep_coeff(const Expr* coeff, const Expr* factors);
  std::pair<const Expr*, const Expr*> as_coeff_Mul(const Expr* e);
  c10::SmallVector<const Expr*, 2> real_roots(const Expr* p, const Expr* x);
  static Num as_num(const Expr* e);
  Tri eval_fact(const Expr* e, Fact f);
  static FactKB default_kb(const Expr* e);

  struct KeyHash {
    size_t operator()(const Expr* e) const {
      return e->hash;
    }
  };
  struct KeyEq {
    bool operator()(const Expr* a, const Expr* b) const;
  };

  std::deque<Expr> storage_;
  std::unordered_set<const Expr*, KeyHash, KeyEq> table_;
  std::vector<SymbolInfo> symbols_;
  std::unordered_map<std::string, std::vector<uint32_t>> symbols_by_name_;
  const Expr* zero_;
  const Expr* one_;
  const Expr* neg_one_;
  const Expr* int_oo_;
  const Expr* neg_int_oo_;
  const Expr* true_;
  const Expr* false_;
  // sympy.core.exprtools._eps, a Dummy(positive=True).
  const Expr* eps_;
  uint64_t dummy_count_ = 0;
  std::unordered_map<const Expr*, SortKeyPtr> sort_keys_;
};

} // namespace torch::symbolic
