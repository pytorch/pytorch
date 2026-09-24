#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>

#include <array>
#include <cstdint>
#include <deque>
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
    return (true_mask & b) ? Tri::True
        : (false_mask & b) ? Tri::False
                           : Tri::Unknown;
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
  // Pow: base, exponent.
  c10::SmallVector<const Expr*, 3> args;
  // Assumptions cache, like sympy's obj._assumptions.
  mutable FactKB kb;

  bool is_number() const {
    return kind <= Kind::NegativeIntInfinity;
  }
  bool is_rational() const {
    return kind == Kind::Integer || kind == Kind::Rational;
  }
};

struct Num {
  int64_t p;
  int64_t q;
};

struct SymbolInfo {
  std::string name;
  // The deduced assumptions (sympy's assumptions0).
  FactKB facts;
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
  const Expr* symbol(const std::string& name, const Facts& facts);
  const SymbolInfo& symbol_info(const Expr* e) const {
    return symbols_.at(e->p);
  }

  // Add(*args), Mul(*args), Pow(b, e), -a and a - b with sympy's evaluation.
  const Expr* add(c10::ArrayRef<const Expr*> args);
  const Expr* mul(c10::ArrayRef<const Expr*> args);
  const Expr* pow(const Expr* b, const Expr* e);
  const Expr* neg(const Expr* a);
  const Expr* sub(const Expr* a, const Expr* b);

  // expr.is_<fact>, ported from sympy's _ask and the _eval_is_* handlers.
  Tri ask(const Expr* e, Fact f);
  // Basic.compare.
  int compare(const Expr* a, const Expr* b) const;

  size_t size() const {
    return storage_.size();
  }

 private:
  const Expr* number(Num n);
  const Expr* intern(
      Kind kind,
      int64_t p,
      int64_t q,
      c10::ArrayRef<const Expr*> args);
  // Assoc node from already-processed args, like AssocOp._from_args.
  const Expr* from_args(Kind kind, c10::SmallVectorImpl<const Expr*>& args);
  const Expr* number_pow(Num b, int64_t e);
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
};

} // namespace torch::symbolic
