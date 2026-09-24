#include <torch/csrc/symbolic/Expr.h>

#include <c10/util/hash.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>

namespace torch::symbolic {

namespace {

using i128 = __int128;

template <typename T>
int cmp3(T a, T b) {
  return (a > b) - (a < b);
}

constexpr int kUnlisted = 1000;

// Index in sympy's ordering_of_classes; classes missing from it sort after all
// listed ones, by class name.
int class_rank(const Expr* e) {
  switch (e->kind) {
    case Kind::Integer:
      return e->p == 0 ? 0 : e->p == 1 ? 1 : e->p == -1 ? 5 : 7;
    case Kind::Rational:
      return e->p == 1 && e->q == 2 ? 2 : 8;
    case Kind::Float:
      return 9;
    case Kind::Symbol:
      return 13;
    case Kind::Pow:
      return 15;
    case Kind::Mul:
      return 16;
    case Kind::Add:
      return 17;
    case Kind::Eq:
      return 62;
    case Kind::Ne:
      return 63;
    case Kind::Gt:
      return 64;
    case Kind::Lt:
      return 65;
    case Kind::Ge:
      return 66;
    case Kind::Le:
      return 67;
    default:
      return kUnlisted;
  }
}

const char* unlisted_class_name(const Expr* e) {
  switch (e->kind) {
    case Kind::Symbol:
      return "Dummy";
    case Kind::IntInfinity:
      return "IntInfinity";
    case Kind::NegativeIntInfinity:
      return "NegativeIntInfinity";
    case Kind::BooleanTrue:
      return "BooleanTrue";
    case Kind::BooleanFalse:
      return "BooleanFalse";
    case Kind::Not:
      return "Not";
    case Kind::And:
      return "And";
    case Kind::Or:
      return "Or";
    default:
      return function_name(e->kind);
  }
}

const std::array<Fact, kNumFacts>& facts_by_name() {
  static const auto order = [] {
    std::array<Fact, kNumFacts> r{};
    for (size_t i = 0; i < kNumFacts; ++i) {
      r[i] = static_cast<Fact>(i);
    }
    std::sort(r.begin(), r.end(), [](Fact a, Fact b) {
      return std::strcmp(fact_name(a), fact_name(b)) < 0;
    });
    return r;
  }();
  return order;
}

i128 gcd128(i128 a, i128 b) {
  if (a < 0) {
    a = -a;
  }
  if (b < 0) {
    b = -b;
  }
  while (b != 0) {
    i128 t = a % b;
    a = b;
    b = t;
  }
  return a;
}

bool fits_int64(i128 v) {
  return v >= std::numeric_limits<int64_t>::min() &&
      v <= std::numeric_limits<int64_t>::max();
}

// Rational p/q in lowest terms with q > 0, like sympy's Rational(p, q).
Num make_num(i128 p, i128 q) {
  if (q < 0) {
    p = -p;
    q = -q;
  }
  i128 g = gcd128(p, q);
  p /= g;
  q /= g;
  if (!fits_int64(p) || !fits_int64(q)) {
    throw NativeUnsupported("integer overflow");
  }
  return {static_cast<int64_t>(p), static_cast<int64_t>(q)};
}

int64_t checked_mul(int64_t a, int64_t b) {
  int64_t r = 0;
  if (__builtin_mul_overflow(a, b, &r)) {
    throw NativeUnsupported("integer overflow");
  }
  return r;
}

// A Number during Add/Mul flattening: the Rational r, or the Float f.
struct NumVal {
  Num r;
  bool is_float = false;
  double f = 0;
};

NumVal num_val(const Expr* e) {
  if (e->kind == Kind::Float) {
    return {{0, 0}, true, e->float_value()};
  }
  if (!e->is_rational()) {
    throw NativeUnsupported("expected a Rational or a Float");
  }
  return {{e->p, e->q}};
}

bool is_zero(const NumVal& n) {
  return n.is_float ? n.f == 0 : n.r.p == 0;
}

// other._as_mpf_op(53) of Float's arithmetic: Integers round to nearest even
// like the int64 conversion; p/q of doubles is rounded once only when both are
// exact.
double float_operand(const NumVal& n) {
  if (n.is_float) {
    return n.f;
  }
  if (n.r.q == 1) {
    return static_cast<double>(n.r.p);
  }
  constexpr int64_t kExact = int64_t(1) << 53;
  if (n.r.p < -kExact || n.r.p > kExact || n.r.q > kExact) {
    throw NativeUnsupported("inexact Rational operand of a Float");
  }
  return static_cast<double>(n.r.p) / static_cast<double>(n.r.q);
}

// Float._new(mpf result, 53). mpmath rounds like IEEE doubles, but with an
// unbounded exponent, so the results agree only when finite and normal.
// A zero result is S.Zero.
NumVal float_result(double v) {
  if (v == 0) {
    return {{0, 1}};
  }
  if (!std::isfinite(v) || std::abs(v) < std::numeric_limits<double>::min()) {
    throw NativeUnsupported("Float result outside the normal doubles");
  }
  return {{0, 0}, true, v};
}

NumVal num_add(const NumVal& a, const NumVal& b) {
  if (a.is_float || b.is_float) {
    return float_result(float_operand(a) + float_operand(b));
  }
  return {
      make_num(i128(a.r.p) * b.r.q + i128(b.r.p) * a.r.q, i128(a.r.q) * b.r.q)};
}

NumVal num_mul(const NumVal& a, const NumVal& b) {
  if (a.is_float || b.is_float) {
    if (is_zero(a) || is_zero(b)) {
      return {{0, 1}};
    }
    double v = float_operand(a) * float_operand(b);
    if (v == 0) {
      throw NativeUnsupported("Float underflow");
    }
    return float_result(v);
  }
  return {make_num(i128(a.r.p) * b.r.p, i128(a.r.q) * b.r.q)};
}

// A Float's mpf tuple (sign, man, exp, bc), man odd; (0, 0, 0, 0) for zero.
std::tuple<int, uint64_t, int64_t, int> float_mpf(double v) {
  if (v == 0) {
    return {0, 0, 0, 0};
  }
  int e = 0;
  auto man = static_cast<uint64_t>(std::ldexp(std::frexp(std::abs(v), &e), 53));
  int tz = std::countr_zero(man);
  man >>= tz;
  return {v < 0, man, int64_t(e) - 53 + tz, std::bit_width(man)};
}

} // namespace

const char* fact_name(Fact f) {
  static constexpr std::array<const char*, kNumFacts> names = {
      "commutative",
      "integer",
      "noninteger",
      "rational",
      "irrational",
      "real",
      "extended_real",
      "finite",
      "infinite",
      "zero",
      "nonzero",
      "positive",
      "negative",
      "nonnegative",
      "nonpositive",
      "extended_positive",
      "extended_negative",
      "extended_nonnegative",
      "extended_nonpositive",
      "extended_nonzero",
      "even",
      "odd",
      "prime",
      "composite",
      "algebraic",
      "transcendental",
      "complex",
      "imaginary",
      "hermitian",
      "antihermitian",
      "polar",
  };
  return names.at(static_cast<size_t>(f));
}

bool ExprArena::KeyEq::operator()(const Expr* a, const Expr* b) const {
  return a->kind == b->kind && a->p == b->p && a->q == b->q &&
      a->args.size() == b->args.size() &&
      std::equal(a->args.begin(), a->args.end(), b->args.begin());
}

ExprArena::ExprArena() {
  zero_ = integer(0);
  one_ = integer(1);
  neg_one_ = integer(-1);
  int_oo_ = intern(Kind::IntInfinity, 0, 0, {});
  neg_int_oo_ = intern(Kind::NegativeIntInfinity, 0, 0, {});
  true_ = intern(Kind::BooleanTrue, 0, 0, {});
  false_ = intern(Kind::BooleanFalse, 0, 0, {});
  eps_ = dummy("_eps", Fact::positive);
}

const Expr* ExprArena::intern(
    Kind kind,
    int64_t p,
    int64_t q,
    c10::ArrayRef<const Expr*> args) {
  Expr key{kind, 0, 0, p, q, {}, {}};
  key.args.assign(args.begin(), args.end());
  key.has_float =
      kind == Kind::Float || std::any_of(args.begin(), args.end(), [](auto a) {
        return a->has_float;
      });
  size_t h = c10::get_hash(static_cast<int>(kind), p, q);
  for (const Expr* a : args) {
    h = c10::hash_combine(h, a->id);
  }
  key.hash = h;
  auto it = table_.find(&key);
  if (it != table_.end()) {
    return *it;
  }
  key.id = static_cast<uint32_t>(storage_.size());
  key.kb = kind == Kind::Symbol ? symbols_.at(p).facts : default_kb(&key);
  const Expr* e = &storage_.emplace_back(std::move(key));
  table_.insert(e);
  return e;
}

const Expr* ExprArena::integer(int64_t v) {
  return intern(Kind::Integer, v, 1, {});
}

const Expr* ExprArena::rational(int64_t p, int64_t q) {
  if (q == 0) {
    throw NativeUnsupported("Rational with zero denominator");
  }
  return number(make_num(p, q));
}

const Expr* ExprArena::float_number(double v) {
  if (!std::isfinite(v)) {
    throw NativeUnsupported("Float of inf or nan");
  }
  // Float(-0.0) is Float(0.0).
  return intern(Kind::Float, std::bit_cast<int64_t>(v == 0 ? 0.0 : v), 0, {});
}

const Expr* ExprArena::number(Num n) {
  if (n.q == 1) {
    return integer(n.p);
  }
  return intern(Kind::Rational, n.p, n.q, {});
}

Num ExprArena::as_num(const Expr* e) {
  if (!e->is_rational()) {
    throw NativeUnsupported("expected a Rational");
  }
  return {e->p, e->q};
}

const Expr* ExprArena::symbol(const std::string& name, const Facts& facts) {
  // Symbol._canonical_assumptions: commutative defaults to True.
  c10::SmallVector<std::pair<Fact, bool>, kNumFacts> given;
  if (facts[static_cast<size_t>(Fact::commutative)] == Tri::Unknown) {
    given.emplace_back(Fact::commutative, true);
  }
  for (size_t i = 0; i < kNumFacts; ++i) {
    if (facts[i] != Tri::Unknown) {
      given.emplace_back(static_cast<Fact>(i), facts[i] == Tri::True);
    }
  }
  FactKB kb;
  deduce_all_facts(kb, given);
  auto& same_name = symbols_by_name_[name];
  for (uint32_t idx : same_name) {
    const FactKB& other = symbols_[idx].facts;
    if (other.true_mask == kb.true_mask && other.false_mask == kb.false_mask) {
      return intern(Kind::Symbol, idx, 0, {});
    }
  }
  const Expr* e = new_symbol(name, kb);
  same_name.push_back(static_cast<uint32_t>(e->p));
  return e;
}

const Expr* ExprArena::dummy(const std::string& name, Fact fact) {
  FactKB kb;
  std::pair<Fact, bool> given[] = {{Fact::commutative, true}, {fact, true}};
  deduce_all_facts(kb, given);
  const Expr* e = new_symbol(name, kb);
  symbols_.back().dummy_index = ++dummy_count_;
  return e;
}

const Expr* ExprArena::new_symbol(const std::string& name, const FactKB& kb) {
  // A zero symbol has an infinite reciprocal, which Mul.flatten turns into nan
  // when multiplied by 0.
  if (kb.get(Fact::commutative) != Tri::True ||
      kb.get(Fact::finite) != Tri::True || kb.get(Fact::zero) == Tri::True) {
    throw NativeUnsupported(
        "only commutative finite nonzero symbols are supported: " + name);
  }
  auto idx = static_cast<int64_t>(symbols_.size());
  symbols_.push_back({name, kb});
  return intern(Kind::Symbol, idx, 0, {});
}

const Expr* ExprArena::from_args(
    Kind kind,
    c10::SmallVectorImpl<const Expr*>& args) {
  if (args.empty()) {
    return kind == Kind::Add ? zero_ : one_;
  }
  if (args.size() == 1) {
    return args[0];
  }
  return intern(kind, 0, 0, args);
}

const Expr* ExprArena::sorted_from_args(
    Kind kind,
    c10::SmallVectorImpl<const Expr*>& args) {
  auto first = args.begin();
  if (!args.empty() && args[0]->is_number()) {
    ++first;
  }
  std::stable_sort(first, args.end(), [&](const Expr* a, const Expr* b) {
    return compare(a, b) < 0;
  });
  return from_args(kind, args);
}

const Expr* ExprArena::number_pow(Num b, int64_t e) {
  if (e == 0) {
    return one_;
  }
  if (e < 0) {
    if (b.p == 0) {
      throw NativeUnsupported("zoo");
    }
    b = make_num(b.q, b.p);
    if (e == std::numeric_limits<int64_t>::min()) {
      if (b.q != 1 || (b.p != 1 && b.p != -1)) {
        throw NativeUnsupported("integer overflow");
      }
      return one_;
    }
    e = -e;
  }
  if (b.q == 1 && (b.p == 0 || b.p == 1)) {
    return integer(b.p);
  }
  if (b.q == 1 && b.p == -1) {
    return integer(e % 2 == 0 ? 1 : -1);
  }
  int64_t p = 1;
  int64_t q = 1;
  for (int64_t i = 0; i < e; ++i) {
    p = checked_mul(p, b.p);
    q = checked_mul(q, b.q);
  }
  return number({p, q});
}

const Expr* ExprArena::float_pow(const Expr* b, int64_t e) {
  // Pow.__new__ and Float._eval_power for e not in {0, 1}. Negating the base
  // for an even e, or the result for an odd one, is exact.
  double x = b->float_value();
  if (x == 0) {
    if (e > 0) {
      return b;
    }
    throw NativeUnsupported("zoo");
  }
  // mpf_pow_int rounds x*x and 1/x once, like IEEE; other powers round the
  // exact power.
  if (e == 2) {
    NumVal n = num_val(b);
    return float_number(num_mul(n, n).f);
  }
  if (e == -1) {
    return float_number(float_result(1 / x).f);
  }
  throw NativeUnsupported("Float power");
}

const Expr* ExprArena::add(c10::ArrayRef<const Expr*> in) {
  // Add.flatten (sympy/core/add.py), restricted to finite commutative terms.
  c10::SmallVector<const Expr*, 8> seq;
  for (const Expr* a : in) {
    if (a->is_boolean()) {
      throw NativeUnsupported("Boolean in Add");
    }
    if (a != zero_) {
      seq.push_back(a);
    }
  }
  if (seq.empty()) {
    return zero_;
  }
  if (seq.size() == 1) {
    return seq[0];
  }
  auto to_expr = [this](const NumVal& n) {
    return n.is_float ? float_number(n.f) : number(n.r);
  };
  NumVal coeff{{0, 1}};
  c10::SmallVector<std::pair<const Expr*, NumVal>, 8> terms;
  std::unordered_map<const Expr*, size_t> term_index;
  for (size_t i = 0; i < seq.size(); ++i) {
    const Expr* o = seq[i];
    const Expr* c = one_;
    const Expr* s = o;
    switch (o->kind) {
      case Kind::Integer:
      case Kind::Rational:
      case Kind::Float:
        coeff = num_add(coeff, num_val(o));
        continue;
      case Kind::IntInfinity:
      case Kind::NegativeIntInfinity:
        throw NativeUnsupported("int_oo in Add");
      case Kind::Add:
        seq.append(o->args.begin(), o->args.end());
        continue;
      case Kind::Mul:
        if (o->args[0]->is_number()) {
          c = o->args[0];
          if (o->args.size() == 2) {
            s = o->args[1];
          } else {
            s = intern(
                Kind::Mul, 0, 0, c10::ArrayRef<const Expr*>(o->args).slice(1));
          }
        }
        break;
      default:
        break;
    }
    NumVal cn = num_val(c);
    auto it = term_index.find(s);
    if (it == term_index.end()) {
      term_index.emplace(s, terms.size());
      terms.emplace_back(s, cn);
    } else {
      NumVal& acc = terms[it->second].second;
      acc = num_add(acc, cn);
    }
  }
  c10::SmallVector<const Expr*, 8> newseq;
  for (const auto& [s, c] : terms) {
    if (is_zero(c)) {
      continue;
    }
    if (!c.is_float && c.r.p == 1 && c.r.q == 1) {
      newseq.push_back(s);
      continue;
    }
    const Expr* ce = to_expr(c);
    if (s->kind == Kind::Mul) {
      c10::SmallVector<const Expr*, 4> margs{ce};
      margs.append(s->args.begin(), s->args.end());
      newseq.push_back(intern(Kind::Mul, 0, 0, margs));
    } else if (s->kind == Kind::Add) {
      throw NativeUnsupported("unevaluated Mul(c, Add)");
    } else {
      newseq.push_back(intern(Kind::Mul, 0, 0, {ce, s}));
    }
  }
  std::sort(newseq.begin(), newseq.end(), [this](auto a, auto b) {
    return compare(a, b) < 0;
  });
  if (!is_zero(coeff)) {
    newseq.insert(newseq.begin(), to_expr(coeff));
  }
  return from_args(Kind::Add, newseq);
}

const Expr* ExprArena::mul(c10::ArrayRef<const Expr*> in) {
  // Mul.flatten (sympy/core/mul.py), restricted to finite commutative factors
  // with Integer exponents.
  c10::SmallVector<const Expr*, 8> seq;
  for (const Expr* a : in) {
    if (a->is_boolean()) {
      throw NativeUnsupported("Boolean in Mul");
    }
    if (a != one_) {
      seq.push_back(a);
    }
  }
  if (seq.empty()) {
    return one_;
  }
  if (seq.size() == 1) {
    return seq[0];
  }
  if (seq.size() == 2) {
    const Expr* a = seq[0];
    const Expr* b = seq[1];
    if (b->is_rational()) {
      std::swap(a, b);
    }
    if (a->is_rational() && a != zero_) {
      const Expr* r = one_;
      const Expr* rest = b;
      if (b->kind == Kind::Mul && b->args[0]->is_number()) {
        r = b->args[0];
        rest = b->args.size() == 2
            ? b->args[1]
            : intern(
                  Kind::Mul,
                  0,
                  0,
                  c10::ArrayRef<const Expr*>(b->args).slice(1));
      }
      if (rest->kind == Kind::Add) {
        if (r != one_) {
          throw NativeUnsupported("unevaluated Mul(c, Add)");
        }
        c10::SmallVector<const Expr*, 8> terms;
        for (const Expr* bi : rest->args) {
          terms.push_back(mul({a, bi}));
        }
        return add(terms);
      }
    }
  }

  NumVal coeff{{1, 1}};
  c10::SmallVector<std::pair<const Expr*, int64_t>, 8> powers;
  std::unordered_map<const Expr*, size_t> power_index;
  for (size_t i = 0; i < seq.size(); ++i) {
    const Expr* o = seq[i];
    const Expr* b = o;
    int64_t e = 1;
    switch (o->kind) {
      case Kind::Mul:
        seq.append(o->args.begin(), o->args.end());
        continue;
      case Kind::Integer:
      case Kind::Rational:
      case Kind::Float:
        coeff = num_mul(coeff, num_val(o));
        continue;
      case Kind::IntInfinity:
      case Kind::NegativeIntInfinity:
        throw NativeUnsupported("int_oo in Mul");
      case Kind::Pow:
        b = o->args[0];
        e = o->args[1]->p;
        break;
      default:
        break;
    }
    auto it = power_index.find(b);
    if (it == power_index.end()) {
      power_index.emplace(b, powers.size());
      powers.emplace_back(b, e);
    } else {
      int64_t& acc = powers[it->second].second;
      if (__builtin_add_overflow(acc, e, &acc)) {
        throw NativeUnsupported("integer overflow");
      }
    }
  }
  if (is_zero(coeff)) {
    return zero_;
  }
  c10::SmallVector<const Expr*, 8> c_part;
  for (const auto& [b, e] : powers) {
    if (e == 0) {
      continue;
    }
    const Expr* p = e == 1 ? b : pow(b, integer(e));
    if (p->kind != Kind::Pow && p != b) {
      throw NativeUnsupported("Pow in Mul did not stay a Pow");
    }
    c_part.push_back(p);
  }
  std::sort(c_part.begin(), c_part.end(), [this](auto a, auto b) {
    return compare(a, b) < 0;
  });
  if (coeff.is_float || !(coeff.r.p == 1 && coeff.r.q == 1)) {
    const Expr* c = coeff.is_float ? float_number(coeff.f) : number(coeff.r);
    if (c_part.size() == 1 && c_part[0]->kind == Kind::Add) {
      c10::SmallVector<const Expr*, 8> terms;
      for (const Expr* f : c_part[0]->args) {
        terms.push_back(mul({c, f}));
      }
      return add(terms);
    }
    c_part.insert(c_part.begin(), c);
  }
  return from_args(Kind::Mul, c_part);
}

const Expr* ExprArena::pow(const Expr* b, const Expr* e) {
  // Pow.__new__ (sympy/core/power.py) for an Integer exponent.
  if (b->is_boolean()) {
    throw NativeUnsupported("Boolean in Pow");
  }
  if (e->kind != Kind::Integer) {
    throw NativeUnsupported("Pow with a non-Integer exponent");
  }
  if (e->p == 0) {
    return one_;
  }
  if (e->p == 1) {
    return b;
  }
  switch (b->kind) {
    case Kind::Integer:
    case Kind::Rational:
      return number_pow(as_num(b), e->p);
    case Kind::Float:
      return float_pow(b, e->p);
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
      throw NativeUnsupported("int_oo in Pow");
    case Kind::Mul: {
      // Mul._eval_power with an Integer exponent.
      c10::SmallVector<const Expr*, 8> factors;
      for (const Expr* f : b->args) {
        factors.push_back(pow(f, e));
      }
      return mul(factors);
    }
    case Kind::Pow:
      // Pow._eval_power with an integer exponent.
      return pow(b->args[0], integer(checked_mul(b->args[1]->p, e->p)));
    default:
      return intern(Kind::Pow, 0, 0, {b, e});
  }
}

int ExprArena::compare(const Expr* a, const Expr* b) const {
  if (a == b) {
    return 0;
  }
  auto rank = [this](const Expr* e) {
    return e->kind == Kind::Symbol && symbol_info(e).dummy_index != 0
        ? kUnlisted
        : class_rank(e);
  };
  int ra = rank(a);
  int c = cmp3(ra, rank(b));
  if (c == 0 && ra == kUnlisted) {
    c = cmp3(std::strcmp(unlisted_class_name(a), unlisted_class_name(b)), 0);
  }
  if (c != 0) {
    return c;
  }
  switch (a->kind) {
    case Kind::Integer:
    case Kind::Rational:
      c = cmp3(a->p, b->p);
      return c != 0 ? c : cmp3(a->q, b->q);
    case Kind::Float: {
      // _hashable_content is (_mpf_, _prec) and the precisions are equal.
      auto ma = float_mpf(a->float_value());
      auto mb = float_mpf(b->float_value());
      return (ma > mb) - (ma < mb);
    }
    case Kind::Symbol: {
      // _hashable_content is (name,) + tuple(sorted(assumptions0.items())).
      const FactKB& fa = symbol_info(a).facts;
      const FactKB& fb = symbol_info(b).facts;
      int n = std::popcount(fa.known);
      c = cmp3(n, std::popcount(fb.known));
      if (c != 0) {
        return c;
      }
      c = cmp3(symbol_info(a).name.compare(symbol_info(b).name), 0);
      if (c != 0) {
        return c;
      }
      const auto& order = facts_by_name();
      auto next = [&order](const FactKB& kb, size_t& i) {
        while (!(kb.known & (1u << static_cast<unsigned>(order[i])))) {
          ++i;
        }
        return order[i++];
      };
      size_t i = 0;
      size_t j = 0;
      for (; n > 0; --n) {
        Fact fact_a = next(fa, i);
        Fact fact_b = next(fb, j);
        if (fact_a != fact_b) {
          return cmp3(std::strcmp(fact_name(fact_a), fact_name(fact_b)), 0);
        }
        c = cmp3(fa.get(fact_a) == Tri::True, fb.get(fact_b) == Tri::True);
        if (c != 0) {
          return c;
        }
      }
      // Dummy._hashable_content appends dummy_index.
      return cmp3(symbol_info(a).dummy_index, symbol_info(b).dummy_index);
    }
    default:
      // _hashable_content is args.
      c = cmp3(a->args.size(), b->args.size());
      for (size_t i = 0; c == 0 && i < a->args.size(); ++i) {
        c = compare(a->args[i], b->args[i]);
      }
      return c;
  }
}

const Expr* ExprArena::neg(const Expr* a) {
  // IntInfinity.__neg__ and NegativeIntInfinity.__neg__.
  if (a == int_oo_ || a == neg_int_oo_) {
    return a == int_oo_ ? neg_int_oo_ : int_oo_;
  }
  // Float.__neg__ keeps a zero a Float.
  if (a->kind == Kind::Float) {
    return float_number(-a->float_value());
  }
  return mul({neg_one_, a});
}

const Expr* ExprArena::sub(const Expr* a, const Expr* b) {
  // Number.__sub__ with a Float skips Add, which would drop a Float zero.
  if (a->is_number() && b->is_number() &&
      (a->kind == Kind::Float || b->kind == Kind::Float)) {
    NumVal r = num_add(num_val(a), num_val(neg(b)));
    return r.is_float ? float_number(r.f) : number(r.r);
  }
  return add({a, neg(b)});
}

} // namespace torch::symbolic
