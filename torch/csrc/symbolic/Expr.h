#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>

#include <array>
#include <bit>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
  Float,
  IntInfinity,
  NegativeIntInfinity,
  Infinity,
  NegativeInfinity,
  Symbol,
  Pow,
  Mul,
  Add,
  // Custom functions from torch/utils/_sympy/functions.py (Functions.cpp).
  Mod,
  PythonMod,
  FloorDiv,
  CleanDiv,
  Max,
  Min,
  PowByNatural,
  FloatPow,
  FloatTrueDiv,
  IntTrueDiv,
  CeilToInt,
  FloorToInt,
  TruncToInt,
  RoundToInt,
  RoundDecimal,
  ToFloat,
  TruncToFloat,
  IsNonOverlappingAndDenseIndicator,
  ModularIndexing,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  Identity,
  Where,
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
  // Integer: value in p. Rational: p/q in lowest terms, q > 1. Float: the
  // bits of a finite double other than -0.0 in p (a sympy Float of precision
  // 53). Symbol: index into ExprArena's symbol table in p.
  int64_t p;
  int64_t q;
  // Add/Mul: sympy's order (coefficient first, the rest by Basic.compare).
  // Pow: base, exponent. Relationals: lhs, rhs. And/Or: sympy's ordered().
  c10::SmallVector<const Expr*, 3> args;
  // Assumptions cache, like sympy's obj._assumptions.
  mutable FactKB kb;
  // A Float, oo or -oo occurs in the expression.
  bool has_float = false;

  bool is_number() const {
    return kind <= Kind::NegativeInfinity;
  }
  bool is_rational() const {
    return kind == Kind::Integer || kind == Kind::Rational;
  }
  double float_value() const {
    return std::bit_cast<double>(p);
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

// Three-way numeric comparison of two Numbers, int_oo- and oo-aware.
int compare_numbers(const Expr* a, const Expr* b);

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
  // Float(v) for a Python float v; throws for inf and nan.
  const Expr* float_number(double v);
  const Expr* int_oo() const {
    return int_oo_;
  }
  const Expr* neg_int_oo() const {
    return neg_int_oo_;
  }
  const Expr* boolean(bool v) const {
    return v ? true_ : false_;
  }
  // sympy.oo and -sympy.oo.
  const Expr* oo() const {
    return oo_;
  }
  const Expr* neg_oo() const {
    return neg_oo_;
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
  // Max(*args, evaluate=evaluate) and Min(*args, evaluate=evaluate).
  const Expr* minmax(Kind kind, c10::ArrayRef<const Expr*> args, bool evaluate = true);
  // CeilDiv(base, divisor), which is always a FloorDiv or a CleanDiv.
  const Expr* ceildiv(const Expr* base, const Expr* divisor);
  // LShift(base, shift) and RShift(base, shift), which always evaluate.
  const Expr* lshift(const Expr* base, const Expr* shift);
  const Expr* rshift(const Expr* base, const Expr* shift);

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
  // ordered(frozenset(seq)); throws when the order would depend on the hash
  // order sympy iterates the frozenset in.
  std::vector<const Expr*> ordered_frozenset(c10::ArrayRef<const Expr*> seq);
  c10::SmallVector<const Expr*, 4> as_ordered_terms(const Expr* e);
  c10::SmallVector<const Expr*, 4> as_ordered_factors(const Expr* e);
  bool could_extract_minus_sign(const Expr* e);
  // Relational.canonical.
  const Expr* canonical(const Expr* r);

  // str(e): sympy's StrPrinter (Printer.cpp).
  std::string str(const Expr* e);

  // Ports of the sympy helpers behind the Add sign handlers (ExprTools.cpp).
  std::pair<const Expr*, const Expr*> as_coeff_Add(const Expr* e);
  std::pair<const Expr*, const Expr*> as_coeff_Mul(const Expr* e);
  c10::SmallVector<const Expr*, 4> free_symbols(const Expr* e) const;
  bool is_polynomial(const Expr* e) const;
  const Expr* diff(const Expr* e, const Expr* x);
  const Expr* xreplace(const Expr* e, c10::ArrayRef<std::pair<const Expr*, const Expr*>> reps);
  std::pair<const Expr*, const Expr*> as_numer_denom(const Expr* e);
  // sympy.core.exprtools._monotonic_sign; nullptr for None.
  const Expr* monotonic_sign(const Expr* e);

  // symbolic_shapes.safe_expand (Expand.cpp).
  const Expr* safe_expand(const Expr* e);
  // symbolic_shapes.canonicalize_bool_expr.
  const Expr* canonicalize_bool_expr(const Expr* e);

  size_t size() const {
    return storage_.size();
  }

 private:
  friend class StrPrinter;

  const Expr* number(Num n);
  const Expr* intern(Kind kind, int64_t p, int64_t q, c10::ArrayRef<const Expr*> args);
  // Assoc node from already-processed args, like AssocOp._from_args.
  const Expr* from_args(Kind kind, c10::SmallVectorImpl<const Expr*>& args);
  // symbolic_shapes._sympy_from_args(sort=True).
  const Expr* sorted_from_args(Kind kind, c10::SmallVectorImpl<const Expr*>& args);
  // expr.func(*args).
  const Expr* rebuild(const Expr* e, c10::ArrayRef<const Expr*> args);
  const Expr* fast_expand(const Expr* e);
  const Expr* expand_multinomial(const Expr* e);
  std::pair<const Expr*, bool> expandsums(c10::ArrayRef<const Expr*> args);
  const Expr* to_nnf(const Expr* e);
  const Expr* lattice_to_nnf(Kind kind, c10::ArrayRef<const Expr*> args);
  const Expr* distribute_and_over_or(const Expr* e);
  const Expr* canonicalize_bool_expr_impl(const Expr* e);
  const Expr* reduce_to_lowest_terms(const Expr* e);
  const Expr* number_pow(Num b, int64_t e);
  const Expr* float_pow(const Expr* b, int64_t e);
  // Mod.eval and PythonMod.eval; nullptr for None.
  const Expr* eval_mod(Kind kind, const Expr* p, const Expr* q);
  // p % q of Numbers, one of them a Float.
  const Expr* eval_float_mod(const Expr* p, const Expr* q);
  // FloorDiv.eval, also for CleanDiv; nullptr for None.
  const Expr* eval_floordiv(const Expr* base, const Expr* divisor);
  // PowByNatural.eval; nullptr for None.
  const Expr* eval_pow_by_natural(const Expr* base, const Expr* exp);
  // sympy.Float(v) of a Python float.
  const Expr* float_of_double(double v);
  // FloatTrueDiv.eval and IntTrueDiv.eval of Numbers.
  const Expr* eval_true_div(Kind kind, const Expr* base, const Expr* divisor);
  // ToFloat.eval of a Number.
  const Expr* eval_to_float(const Expr* number);
  // CeilToInt/FloorToInt/TruncToInt/RoundToInt.eval; nullptr for None.
  const Expr* eval_to_int(Kind kind, const Expr* number);
  // IsNonOverlappingAndDenseIndicator.eval; nullptr for None.
  const Expr* eval_is_non_overlapping_and_dense(c10::ArrayRef<const Expr*> args);
  // ModularIndexing.eval; nullptr for None.
  const Expr* eval_modular_indexing(const Expr* base, const Expr* divisor, const Expr* modulus);
  // BitwiseFn_bitwise_and/or/xor.eval; nullptr for None.
  const Expr* eval_bitwise(Kind kind, const Expr* a, const Expr* b);
  const Expr* new_symbol(const std::string& name, const FactKB& kb);
  const Expr* as_boolean(const Expr* e);
  // The tail of LatticeOp.__new__.
  const Expr* lattice(Kind kind, c10::ArrayRef<const Expr*> args);
  const Expr* keep_coeff(const Expr* coeff, const Expr* factors);
  c10::SmallVector<const Expr*, 2> real_roots(const Expr* p, const Expr* x);
  static Num as_num(const Expr* e);
  Tri eval_fact(const Expr* e, Fact f);
  static FactKB default_kb(const Expr* e);

  void grow_table();

  // Open-addressing intern table over Expr::hash with linear probing; the
  // capacity is a power of two, 1 << (64 - table_shift_).
  struct Slot {
    size_t hash;
    const Expr* e;
  };

  std::deque<Expr> storage_;
  std::vector<Slot> table_;
  int table_shift_ = 64;
  std::vector<SymbolInfo> symbols_;
  std::unordered_map<std::string, std::vector<uint32_t>> symbols_by_name_;
  const Expr* zero_;
  const Expr* one_;
  const Expr* neg_one_;
  const Expr* int_oo_;
  const Expr* neg_int_oo_;
  const Expr* oo_;
  const Expr* neg_oo_;
  const Expr* true_;
  const Expr* false_;
  // sympy.core.exprtools._eps, a Dummy(positive=True).
  const Expr* eps_;
  uint64_t dummy_count_ = 0;
  std::unordered_map<const Expr*, SortKeyPtr> sort_keys_;
};

} // namespace torch::symbolic
