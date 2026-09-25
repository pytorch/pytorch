#include <torch/csrc/symbolic/ValueRanges.h>

#include <c10/util/SmallVector.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

// SymPyValueRangeAnalysis and sympy_interp (torch/utils/_sympy/interp.py).
// Bounds are compared and combined with the Python operators on sympy numbers,
// including int_oo's overloads; a nan bound throws like simple_sympify.

namespace torch::symbolic {

namespace {

using i128 = __int128;

bool is_int_oo(const Expr* e) {
  return e->kind == Kind::IntInfinity || e->kind == Kind::NegativeIntInfinity;
}

bool is_infinite(const Expr* e) {
  return is_int_oo(e) || e->kind == Kind::Infinity ||
      e->kind == Kind::NegativeInfinity;
}

int sign(const Expr* e) {
  switch (e->kind) {
    case Kind::IntInfinity:
    case Kind::Infinity:
      return 1;
    case Kind::NegativeIntInfinity:
    case Kind::NegativeInfinity:
      return -1;
    case Kind::Float: {
      double v = e->float_value();
      return (v > 0) - (v < 0);
    }
    default:
      return (e->p > 0) - (e->p < 0);
  }
}

bool lt(const Expr* a, const Expr* b) {
  return compare_numbers(a, b) < 0;
}

// x in r by bounds (Python's x in ValueRanges.unknown_int() is True for any x;
// see true_div).
bool contains(const ValueRanges& r, const Expr* x) {
  return !lt(x, r.lower) && !lt(r.upper, x);
}

// Python's min(a, b) and max(a, b).
const Expr* min_num(const Expr* a, const Expr* b) {
  return lt(b, a) ? b : a;
}

const Expr* max_num(const Expr* a, const Expr* b) {
  return lt(a, b) ? b : a;
}

// int(x) of a finite Number.
int64_t py_int(const Expr* x) {
  switch (x->kind) {
    case Kind::Integer:
      return x->p;
    case Kind::Rational:
      return x->p / x->q;
    case Kind::Float: {
      double t = std::trunc(x->float_value());
      if (!(std::fabs(t) < 0x1p63)) {
        throw NativeUnsupported("int of a large Float");
      }
      return static_cast<int64_t>(t);
    }
    default:
      throw NativeUnsupported("int of an infinity");
  }
}

int bit_length(int64_t n) {
  uint64_t m = n < 0 ? -static_cast<uint64_t>(n) : n;
  return 64 - std::countl_zero(m);
}

void require_int(const ValueRanges& r) {
  if (!r.is_int()) {
    throw NativeUnsupported("non-integer range in an integer handler");
  }
}

void require_not_bool(const ValueRanges& r) {
  if (r.is_bool()) {
    throw NativeUnsupported("bool range in a numeric handler");
  }
}

void require_bool(const ValueRanges& r) {
  if (!r.is_bool()) {
    throw NativeUnsupported("not bool like");
  }
}

class Analysis {
 public:
  Analysis(
      ExprArena& arena,
      const RangeMap& ranges,
      const RangeMap* context_ranges = nullptr)
      : a_(arena), ranges_(ranges), context_ranges_(context_ranges) {}

  ValueRanges interp(const Expr* e);
  // _rewrite_for_value_range_analysis.
  const Expr* rewrite(const Expr* e);

 private:
  using Terms = c10::SmallVector<std::pair<const Expr*, const Expr*>, 8>;

  const Expr* zero() {
    return a_.integer(0);
  }
  const Expr* one() {
    return a_.integer(1);
  }
  ValueRanges unknown() {
    return {a_.neg_oo(), a_.oo()};
  }
  ValueRanges unknown_int() {
    return {a_.neg_int_oo(), a_.int_oo()};
  }
  ValueRanges unknown_bool() {
    return {a_.boolean(false), a_.boolean(true)};
  }
  bool contains_zero(const ValueRanges& r) {
    return contains(r, zero());
  }

  const Expr* keep_float(const Expr* r, const Expr* x, const Expr* y);
  const Expr* num_add(const Expr* x, const Expr* y);
  const Expr* num_sub(const Expr* x, const Expr* y) {
    return num_add(x, a_.neg(y));
  }
  const Expr* num_abs(const Expr* x) {
    return sign(x) < 0 ? a_.neg(x) : x;
  }
  const Expr* safe_mul(const Expr* x, const Expr* y);
  const Expr* safe_pow(const Expr* base, const Expr* exp);
  const Expr* safe_pow_abs(const Expr* base, int64_t exp);
  const Expr* pow_by_natural_fn(const Expr* base, const Expr* exp);

  ValueRanges default_symbol_range(const Expr* s);
  ValueRanges not_(const ValueRanges& x);
  ValueRanges and_(const ValueRanges& x, const ValueRanges& y);
  ValueRanges or_(const ValueRanges& x, const ValueRanges& y);
  ValueRanges bool_to_int(const ValueRanges& x);
  ValueRanges eq(ValueRanges x, ValueRanges y);
  ValueRanges lt_(const ValueRanges& x, const ValueRanges& y);
  ValueRanges add(const ValueRanges& x, const ValueRanges& y);
  ValueRanges mul(const ValueRanges& x, const ValueRanges& y);
  ValueRanges floordiv(const ValueRanges& x, const ValueRanges& y);
  ValueRanges mod(const ValueRanges& x, const ValueRanges& y);
  ValueRanges python_mod(const ValueRanges& x, const ValueRanges& y);
  ValueRanges pow_by_natural(const ValueRanges& x, const ValueRanges& y);
  ValueRanges min_or_max(Kind kind, const ValueRanges& x, const ValueRanges& y);
  ValueRanges bitwise(Kind kind, ValueRanges x, ValueRanges y);
  ValueRanges true_div(Kind kind, const ValueRanges& x, const ValueRanges& y);
  const Expr* floor_or_ceiling(Kind kind, const Expr* x);

  template <typename F>
  ValueRanges increasing_map(const ValueRanges& x, const F& fn) {
    require_not_bool(x);
    return make_value_range(a_, fn(x.lower), fn(x.upper));
  }
  ValueRanges increasing_map(Kind kind, const ValueRanges& x) {
    return increasing_map(
        x, [&](const Expr* b) { return a_.function(kind, {b}); });
  }
  template <typename F>
  ValueRanges coordinatewise_monotone_map(
      const ValueRanges& x,
      const ValueRanges& y,
      const F& fn);

  const ValueRanges* find_range(const Expr* s) const;
  bool definitely_ge(const Expr* e, int64_t lower);
  Terms terms_of_add(const Expr* e);
  const Expr* rewrite_mod_subtractions_in_add(const Expr* e);

  ExprArena& a_;
  const RangeMap& ranges_;
  const RangeMap* context_ranges_;
};

// functions._keep_float: Float(float(r)) if an operand is a Float and r isn't.
const Expr* Analysis::keep_float(const Expr* r, const Expr* x, const Expr* y) {
  if (r->kind == Kind::Float ||
      (x->kind != Kind::Float && y->kind != Kind::Float)) {
    return r;
  }
  switch (r->kind) {
    case Kind::IntInfinity:
    case Kind::Infinity:
      return a_.oo();
    case Kind::NegativeIntInfinity:
    case Kind::NegativeInfinity:
      return a_.neg_oo();
    default:
      if (r->kind == Kind::Integer && r->p == 0) {
        return a_.float_number(0.0);
      }
      throw NativeUnsupported("keep_float of a finite non-Float");
  }
}

const Expr* Analysis::num_add(const Expr* x, const Expr* y) {
  bool x_inf = is_infinite(x);
  bool y_inf = is_infinite(y);
  if (!x_inf && !y_inf) {
    return a_.add({x, y});
  }
  if (x_inf && y_inf && x != y) {
    if (is_int_oo(x) == is_int_oo(y)) {
      throw NativeUnsupported("sympy expression is NaN");
    }
    // oo and -oo absorb +-int_oo, except that NegativeIntInfinity.__add__
    // returns itself for -oo.
    if (is_int_oo(x) &&
        (x->kind == Kind::IntInfinity || y->kind == Kind::Infinity)) {
      return y;
    }
    return x;
  }
  return x_inf ? x : y;
}

// safe_mul in SymPyValueRangeAnalysis.mul.
const Expr* Analysis::safe_mul(const Expr* x, const Expr* y) {
  if (sign(x) == 0) {
    return x;
  }
  if (sign(y) == 0) {
    return y;
  }
  if (is_infinite(x) || is_infinite(y)) {
    // The infinity's class, x's when both are infinite.
    bool positive = sign(x) * sign(y) > 0;
    if (is_int_oo(is_infinite(x) ? x : y)) {
      return positive ? a_.int_oo() : a_.neg_int_oo();
    }
    return positive ? a_.oo() : a_.neg_oo();
  }
  return a_.mul({x, y});
}

// functions.safe_pow.
const Expr* Analysis::safe_pow(const Expr* base, const Expr* exp) {
  if (exp->kind != Kind::Integer || exp->p < 0) {
    // A negative exponent raises ValueError; +-int_oo recurses forever.
    throw NativeUnsupported("safe_pow needs a natural exponent");
  }
  if (is_int_oo(base)) {
    if (exp->p == 0) {
      return one();
    }
    return base == a_.neg_int_oo() && exp->p % 2 == 1 ? a_.neg_int_oo()
                                                        : a_.int_oo();
  }
  if (base->kind == Kind::Integer) {
    return a_.function(Kind::PowByNatural, {base, exp});
  }
  if (sign(base) < 0) {
    const Expr* r = safe_pow_abs(a_.neg(base), exp->p);
    return exp->p % 2 == 0 ? r : a_.neg(r);
  }
  return safe_pow_abs(base, exp->p);
}

// functions._safe_pow: repeated squaring with sympy products, int_oo past
// sys.maxsize.
const Expr* Analysis::safe_pow_abs(const Expr* base, int64_t exp) {
  if (exp == 0) {
    return one();
  }
  const Expr* half = safe_pow_abs(base, exp / 2);
  if (half == a_.int_oo()) {
    return half;
  }
  const Expr* max_size = a_.integer(std::numeric_limits<int64_t>::max());
  const Expr* r = a_.mul({half, half});
  if (lt(max_size, r)) {
    return a_.int_oo();
  }
  if (exp % 2 == 1) {
    r = a_.mul({r, base});
    if (lt(max_size, r)) {
      return a_.int_oo();
    }
  }
  return r;
}

// PowByNatural(base, exp) for base >= 1 and exp >= 0.
const Expr* Analysis::pow_by_natural_fn(const Expr* base, const Expr* exp) {
  if (base == a_.int_oo() && exp->kind == Kind::Integer) {
    // sympy.Pow(int_oo, exp).
    return exp->p == 0 ? one() : a_.int_oo();
  }
  return a_.function(Kind::PowByNatural, {base, exp});
}

// _default_symbol_range.
ValueRanges Analysis::default_symbol_range(const Expr* s) {
  if (a_.ask(s, Fact::integer) != Tri::True) {
    return unknown();
  }
  if (a_.ask(s, Fact::positive) == Tri::True) {
    return {one(), a_.int_oo()};
  }
  if (a_.ask(s, Fact::nonnegative) == Tri::True) {
    return {zero(), a_.int_oo()};
  }
  return unknown_int();
}

ValueRanges Analysis::not_(const ValueRanges& x) {
  require_bool(x);
  return {a_.logical_not(x.upper), a_.logical_not(x.lower)};
}

ValueRanges Analysis::and_(const ValueRanges& x, const ValueRanges& y) {
  require_bool(x);
  require_bool(y);
  return {
      a_.logical_and({x.lower, y.lower}), a_.logical_and({x.upper, y.upper})};
}

ValueRanges Analysis::or_(const ValueRanges& x, const ValueRanges& y) {
  require_bool(x);
  require_bool(y);
  return {
      a_.logical_or({x.lower, y.lower}), a_.logical_or({x.upper, y.upper})};
}

ValueRanges Analysis::bool_to_int(const ValueRanges& x) {
  if (x.is_singleton()) {
    const Expr* v = a_.integer(x.lower == a_.boolean(true) ? 1 : 0);
    return {v, v};
  }
  return {zero(), one()};
}

ValueRanges Analysis::eq(ValueRanges x, ValueRanges y) {
  if (x.is_singleton() && y.is_singleton() && x.lower == y.lower) {
    return {a_.boolean(true), a_.boolean(true)};
  }
  if (x.is_bool()) {
    x = bool_to_int(x);
  }
  if (y.is_bool()) {
    y = bool_to_int(y);
  }
  if (lt(y.upper, x.lower) || lt(x.upper, y.lower)) {
    return {a_.boolean(false), a_.boolean(false)};
  }
  return unknown_bool();
}

ValueRanges Analysis::lt_(const ValueRanges& x, const ValueRanges& y) {
  if (x.is_bool() != y.is_bool()) {
    throw NativeUnsupported("lt of a bool and a numeric range");
  }
  if (x.is_bool()) {
    return and_(not_(x), y);
  }
  if (lt(x.upper, y.lower)) {
    return {a_.boolean(true), a_.boolean(true)};
  }
  if (!lt(x.lower, y.upper)) {
    return {a_.boolean(false), a_.boolean(false)};
  }
  return unknown_bool();
}

ValueRanges Analysis::add(const ValueRanges& x, const ValueRanges& y) {
  require_not_bool(x);
  require_not_bool(y);
  return make_value_range(
      a_,
      keep_float(num_add(x.lower, y.lower), x.lower, y.lower),
      keep_float(num_add(x.upper, y.upper), x.upper, y.upper));
}

ValueRanges Analysis::mul(const ValueRanges& x, const ValueRanges& y) {
  if (x.is_bool() != y.is_bool()) {
    throw NativeUnsupported("mul of a bool and a numeric range");
  }
  if (x.is_bool()) {
    return and_(x, y);
  }
  auto product = [this](const Expr* p, const Expr* q) {
    return keep_float(safe_mul(p, q), p, q);
  };
  return coordinatewise_monotone_map(x, y, product);
}

template <typename F>
ValueRanges Analysis::coordinatewise_monotone_map(
    const ValueRanges& x,
    const ValueRanges& y,
    const F& fn) {
  std::array<const Expr*, 4> products = {
      fn(x.lower, y.lower),
      fn(x.lower, y.upper),
      fn(x.upper, y.lower),
      fn(x.upper, y.upper)};
  const Expr* lo = products[0];
  const Expr* hi = products[0];
  for (const Expr* p : products) {
    if (!p->is_number()) {
      throw NativeUnsupported("symbolic value range bound");
    }
    lo = min_num(lo, p);
    hi = max_num(hi, p);
  }
  return make_value_range(a_, lo, hi);
}

// int_truediv and truediv.
ValueRanges Analysis::true_div(
    Kind kind,
    const ValueRanges& x,
    const ValueRanges& y) {
  require_not_bool(x);
  require_not_bool(y);
  if (contains(y, zero())) {
    return unknown();
  }
  bool is_int = kind == Kind::IntTrueDiv;
  const Expr* pos = is_int ? a_.int_oo() : a_.oo();
  const Expr* neg = is_int ? a_.neg_int_oo() : a_.neg_oo();
  if (contains(y, neg) || contains(y, pos)) {
    if (contains(x, neg) || contains(x, pos)) {
      return unknown();
    }
    // Python gives unknown() if x is the cached unknown_int() and otherwise
    // raises on a nan product.
    if (x.lower == a_.neg_int_oo() && x.upper == a_.int_oo()) {
      throw NativeUnsupported("membership in unknown_int() is by identity");
    }
  }
  return coordinatewise_monotone_map(x, y, [&](const Expr* p, const Expr* q) {
    return keep_float(a_.function(kind, {p, q}), p, q);
  });
}

// sympy.floor / sympy.ceiling of a number.
const Expr* Analysis::floor_or_ceiling(Kind kind, const Expr* x) {
  bool ceil = kind == Kind::CeilToInt;
  switch (x->kind) {
    case Kind::Rational: {
      i128 p = ceil ? -i128(x->p) : i128(x->p);
      i128 f = p / x->q - (p % x->q < 0);
      return a_.integer(static_cast<int64_t>(ceil ? -f : f));
    }
    case Kind::Float: {
      double v = x->float_value();
      double r = ceil ? std::ceil(v) : std::floor(v);
      if (!(r >= -0x1p63 && r < 0x1p63)) {
        throw NativeUnsupported("integer overflow");
      }
      return a_.integer(static_cast<int64_t>(r));
    }
    default:
      return x;
  }
}

ValueRanges Analysis::floordiv(const ValueRanges& x, const ValueRanges& y) {
  require_not_bool(x);
  require_not_bool(y);
  if (contains_zero(y)) {
    bool x_nonneg = !lt(x.lower, zero());
    bool x_nonpos = !lt(zero(), x.upper);
    bool y_nonneg = !lt(y.lower, zero());
    bool y_nonpos = !lt(zero(), y.upper);
    if ((y_nonneg && x_nonneg) || (y_nonpos && x_nonpos)) {
      return {zero(), a_.int_oo()};
    }
    if ((y_nonpos && x_nonneg) || (y_nonneg && x_nonpos)) {
      return {a_.neg_int_oo(), zero()};
    }
    return unknown_int();
  }
  auto quotient = [this](const Expr* p, const Expr* q) {
    // FloorDiv is nan when both sides are infinite.
    if (is_infinite(p) && is_infinite(q)) {
      return sign(p) * sign(q) > 0 ? a_.int_oo() : a_.neg_int_oo();
    }
    return a_.function(Kind::FloorDiv, {p, q});
  };
  return coordinatewise_monotone_map(x, y, quotient);
}

// C semantics, like SymPyValueRangeAnalysis.mod.
ValueRanges Analysis::mod(const ValueRanges& x, const ValueRanges& y) {
  require_not_bool(x);
  require_not_bool(y);
  if (contains_zero(y)) {
    return unknown_int();
  }
  if (y.is_singleton()) {
    require_int(x);
    require_int(y);
    if (is_int_oo(y.lower)) {
      throw NativeUnsupported("int_oo in Mod");
    }
    i128 y_val = y.lower->p < 0 ? -i128(y.lower->p) : i128(y.lower->p);
    // c_div(a, y_val) truncates; it is oo for int_oo and -int_oo for -int_oo.
    bool same_class = is_int_oo(x.lower) || is_int_oo(x.upper)
        ? x.lower == x.upper
        : i128(x.lower->p) / y_val == i128(x.upper->p) / y_val;
    if (same_class) {
      if (is_int_oo(x.lower)) {
        // c_mod(+-int_oo, y_val) is nan.
        throw NativeUnsupported("sympy expression is NaN");
      }
      return {
          a_.integer(static_cast<int64_t>(i128(x.lower->p) % y_val)),
          a_.integer(static_cast<int64_t>(i128(x.upper->p) % y_val))};
    }
    if (y_val > std::numeric_limits<int64_t>::max()) {
      throw NativeUnsupported("integer overflow");
    }
    const Expr* hi = a_.integer(static_cast<int64_t>(y_val - 1));
    const Expr* lo = a_.integer(static_cast<int64_t>(1 - y_val));
    if (lt(x.upper, zero())) {
      return {lo, zero()};
    }
    if (lt(zero(), x.lower)) {
      return {zero(), hi};
    }
    return {max_num(lo, x.lower), min_num(hi, x.upper)};
  }
  // cls.abs(y).upper - 1; 0 is not in y, so abs is a monotone map.
  const Expr* upper =
      num_sub(max_num(num_abs(y.lower), num_abs(y.upper)), one());
  return make_value_range(a_, a_.neg(upper), upper);
}

ValueRanges Analysis::python_mod(const ValueRanges& x, const ValueRanges& y) {
  require_not_bool(x);
  require_not_bool(y);
  if (!lt(x.lower, zero()) && !lt(y.lower, zero())) {
    return mod(x, y);
  }
  return make_value_range(
      a_,
      lt(y.lower, zero()) ? num_add(y.lower, one()) : zero(),
      lt(zero(), y.upper) ? num_sub(y.upper, one()) : zero());
}

ValueRanges Analysis::pow_by_natural(
    const ValueRanges& x,
    const ValueRanges& y) {
  require_not_bool(x);
  require_int(y);
  if (x.is_singleton() && y.is_singleton()) {
    const Expr* r = safe_pow(x.lower, y.lower);
    return {r, r};
  }
  if (!lt(x.lower, one())) {
    // y & ValueRanges(0, int_oo)
    ValueRanges exp(max_num(y.lower, zero()), y.upper);
    return make_value_range(
        a_,
        pow_by_natural_fn(x.lower, exp.lower),
        pow_by_natural_fn(x.upper, exp.upper));
  }
  if (y.is_singleton()) {
    const Expr* l = safe_pow(x.lower, y.lower);
    const Expr* u = safe_pow(x.upper, y.lower);
    if (y.lower->p % 2 == 1) {
      return {l, u};
    }
    // convex_min_zero_map
    if (contains_zero(x)) {
      const Expr* upper = max_num(l, u);
      bool is_float = upper->kind == Kind::Float;
      return {is_float ? a_.float_number(0.0) : zero(), upper};
    }
    return {min_num(l, u), max_num(l, u)};
  }
  const Expr* r = safe_pow(max_num(x.upper, a_.neg(x.lower)), y.upper);
  return {a_.neg(r), r};
}

ValueRanges Analysis::min_or_max(
    Kind kind,
    const ValueRanges& x,
    const ValueRanges& y) {
  if (x.is_bool() != y.is_bool()) {
    throw NativeUnsupported("min/max of a bool and a numeric range");
  }
  if (x.is_bool()) {
    return kind == Kind::Min ? and_(x, y) : or_(x, y);
  }
  // sympy.Min/Max break ties between different number types by type.
  auto pick = [kind](const Expr* p, const Expr* q) {
    int c = compare_numbers(p, q);
    if (c == 0 && p != q) {
      throw NativeUnsupported("Min/Max of equal numbers of different types");
    }
    return (kind == Kind::Min) == (c < 0) ? p : q;
  };
  return make_value_range(a_, pick(x.lower, y.lower), pick(x.upper, y.upper));
}

// bitwise_and, bitwise_or and bitwise_xor.
ValueRanges Analysis::bitwise(Kind kind, ValueRanges x, ValueRanges y) {
  if (x.is_bool() && y.is_bool()) {
    if (kind == Kind::BitwiseAnd) {
      return and_(x, y);
    }
    if (kind == Kind::BitwiseOr) {
      return or_(x, y);
    }
    bool has_false = false;
    bool has_true = false;
    for (const Expr* p : {x.lower, x.upper}) {
      for (const Expr* q : {y.lower, y.upper}) {
        (p == q ? has_false : has_true) = true;
      }
    }
    return {a_.boolean(!has_false), a_.boolean(has_true)};
  }
  if (x.is_bool()) {
    x = bool_to_int(x);
  }
  if (y.is_bool()) {
    y = bool_to_int(y);
  }
  switch (kind) {
    case Kind::BitwiseAnd: {
      const Expr* lower = min_num(x.lower, y.lower);
      // A lower bound of -oo or -int_oo also gives 0.
      if (lt(lower, zero()) && !is_infinite(lower)) {
        // -(1 << int(-lower - 1).bit_length())
        int64_t n = lower->kind == Kind::Integer ? ~lower->p
            : lower->kind == Kind::Rational
            ? static_cast<int64_t>((-i128(lower->p) - lower->q) / lower->q)
            : py_int(a_.sub(a_.neg(lower), one()));
        lower = a_.integer(static_cast<int64_t>(-(i128(1) << bit_length(n))));
      } else {
        lower = zero();
      }
      return make_value_range(a_, lower, max_num(x.upper, y.upper));
    }
    case Kind::BitwiseOr: {
      const Expr* upper = max_num(x.upper, y.upper);
      if (lt(zero(), upper) && !is_infinite(upper)) {
        // (1 << int(upper).bit_length()) - 1
        upper = a_.integer(
            static_cast<int64_t>((i128(1) << bit_length(py_int(upper))) - 1));
      } else if (lt(upper, zero())) {
        upper = a_.integer(-1);
      }
      return make_value_range(a_, min_num(x.lower, y.lower), upper);
    }
    default:
      if (x.is_singleton() && y.is_singleton() &&
          x.lower->kind == Kind::Integer && y.lower->kind == Kind::Integer) {
        const Expr* v = a_.integer(x.lower->p ^ y.lower->p);
        return {v, v};
      }
      return unknown_int();
  }
}

ValueRanges Analysis::interp(const Expr* e) {
  switch (e->kind) {
    case Kind::Integer:
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
    case Kind::BooleanTrue:
    case Kind::BooleanFalse:
    case Kind::Rational:
    case Kind::Float:
    case Kind::Infinity:
    case Kind::NegativeInfinity:
      return {e, e};
    case Kind::Symbol: {
      const ValueRanges* r = find_range(e);
      return r != nullptr ? *r : default_symbol_range(e);
    }
    default:
      break;
  }
  c10::SmallVector<ValueRanges, 3> args;
  for (const Expr* arg : e->args) {
    args.push_back(interp(arg));
  }
  // The associative handlers fold left over the args.
  auto fold = [&](auto&& f) {
    if (args.size() < 2) {
      throw NativeUnsupported("associative op needs >1 args");
    }
    ValueRanges acc = f(args[0], args[1]);
    for (size_t i = 2; i < args.size(); ++i) {
      acc = f(acc, args[i]);
    }
    return acc;
  };
  auto binary = [this](auto method) {
    return [this, method](const ValueRanges& x, const ValueRanges& y) {
      return (this->*method)(x, y);
    };
  };
  switch (e->kind) {
    case Kind::Add:
      return fold(binary(&Analysis::add));
    case Kind::Mul:
      return fold(binary(&Analysis::mul));
    case Kind::Pow:
      // The arena only builds Integer exponents; a negative one takes the pow
      // handler, which is unknown().
      if (e->args[1]->kind != Kind::Integer) {
        throw NativeUnsupported("pow exponent");
      }
      return e->args[1]->p >= 0 ? pow_by_natural(args[0], args[1]) : unknown();
    case Kind::PowByNatural:
      return pow_by_natural(args[0], args[1]);
    case Kind::Mod:
      return mod(args[0], args[1]);
    case Kind::PythonMod:
      return python_mod(args[0], args[1]);
    case Kind::FloorDiv:
    case Kind::CleanDiv:
      return floordiv(args[0], args[1]);
    case Kind::Max:
    case Kind::Min:
      return fold([&](const ValueRanges& x, const ValueRanges& y) {
        return min_or_max(e->kind, x, y);
      });
    case Kind::CeilToInt:
    case Kind::FloorToInt:
      // increasing_map(sympy.ceiling / sympy.floor), the identity on integer
      // bounds.
      if (args[0].is_int()) {
        return args[0];
      }
      return increasing_map(
          args[0], [&](const Expr* b) { return floor_or_ceiling(e->kind, b); });
    case Kind::TruncToInt:
    case Kind::RoundToInt:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
      return increasing_map(e->kind, args[0]);
    case Kind::IsNonOverlappingAndDenseIndicator:
      return unknown_int();
    case Kind::ModularIndexing:
      return mod(floordiv(args[0], args[1]), args[2]);
    case Kind::BitwiseAnd:
    case Kind::BitwiseOr:
    case Kind::BitwiseXor:
      return bitwise(e->kind, args[0], args[1]);
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
      return true_div(e->kind, args[0], args[1]);
    case Kind::FloatPow:
      return unknown();
    case Kind::RoundDecimal: {
      if (!args[1].is_singleton()) {
        return unknown();
      }
      const Expr* ndigits = args[1].lower;
      return increasing_map(args[0], [&](const Expr* b) {
        return a_.function(Kind::RoundDecimal, {b, ndigits});
      });
    }
    case Kind::Eq:
      return eq(args[0], args[1]);
    case Kind::Ne:
      return not_(eq(args[0], args[1]));
    case Kind::Lt:
      return lt_(args[0], args[1]);
    case Kind::Gt:
      return lt_(args[1], args[0]);
    case Kind::Le:
      return not_(lt_(args[1], args[0]));
    case Kind::Ge:
      return not_(lt_(args[0], args[1]));
    case Kind::Not:
      return not_(args[0]);
    case Kind::And:
      return fold(binary(&Analysis::and_));
    case Kind::Or:
      return fold(binary(&Analysis::or_));
    default:
      throw NativeUnsupported("no value range handler");
  }
}

const ValueRanges* Analysis::find_range(const Expr* s) const {
  if (auto it = ranges_.find(s); it != ranges_.end()) {
    return &it->second;
  }
  if (context_ranges_ != nullptr) {
    if (auto it = context_ranges_->find(s); it != context_ranges_->end()) {
      return &it->second;
    }
  }
  return nullptr;
}

// _definitely_ge. _bound_sympy_for_rewrite_guard's None cases throw here, so
// the caller falls back to Python.
bool Analysis::definitely_ge(const Expr* e, int64_t lower) {
  if (lower == 0 && a_.ask(e, Fact::nonnegative) == Tri::True) {
    return true;
  }
  if (lower == 1 && a_.ask(e, Fact::positive) == Tri::True) {
    return true;
  }
  if (e->kind == Kind::Symbol) {
    if (const ValueRanges* r = find_range(e)) {
      require_not_bool(*r);
      return !lt(r->lower, a_.integer(lower));
    }
  }
  ValueRanges r = interp(e);
  // _definitely_ge_value: a Boolean bound >= lower is a TypeError.
  return !r.is_bool() && !lt(r.lower, a_.integer(lower));
}

Analysis::Terms Analysis::terms_of_add(const Expr* e) {
  Terms terms;
  auto add_term = [&](const Expr* t) {
    auto [coeff, factor] = a_.as_coeff_Mul(t);
    auto it = std::find_if(terms.begin(), terms.end(), [&](const auto& kv) {
      return kv.first == factor;
    });
    if (it == terms.end()) {
      terms.emplace_back(factor, coeff);
    } else {
      it->second = a_.add({it->second, coeff});
    }
  };
  if (e->kind == Kind::Add) {
    for (const Expr* t : e->args) {
      add_term(t);
    }
  } else {
    add_term(e);
  }
  return terms;
}

const Expr* Analysis::rewrite_mod_subtractions_in_add(const Expr* e) {
  Terms terms = terms_of_add(e);
  auto coeff_of = [&](const Expr* factor) -> const Expr** {
    for (auto& [f, c] : terms) {
      if (f == factor) {
        return &c;
      }
    }
    return nullptr;
  };
  c10::SmallVector<const Expr*, 4> replacements;
  const Terms snapshot = terms;
  for (const auto& [factor, mod_coeff] : snapshot) {
    if (mod_coeff->kind != Kind::Integer || mod_coeff->p == 0) {
      continue;
    }
    if (factor->kind != Kind::Mod && factor->kind != Kind::PythonMod) {
      continue;
    }
    const Expr* base = factor->args[0];
    const Expr* divisor = factor->args[1];
    // _mod_rewrite_is_valid
    if (!definitely_ge(divisor, 1) ||
        (factor->kind == Kind::Mod && !definitely_ge(base, 0))) {
      continue;
    }
    Terms matched;
    bool all_matched = true;
    for (const auto& [base_factor, base_coeff] : terms_of_add(base)) {
      if (base_coeff->kind != Kind::Integer) {
        all_matched = false;
        break;
      }
      const Expr** term = coeff_of(base_factor);
      const Expr* term_coeff = term != nullptr ? *term : zero();
      const Expr* needed = a_.mul({a_.neg(mod_coeff), base_coeff});
      int needed_sign = sign(needed);
      if (needed_sign == 0) {
        continue;
      }
      if (sign(term_coeff) * needed_sign <= 0 ||
          (needed_sign > 0 && lt(term_coeff, needed)) ||
          (needed_sign < 0 && lt(needed, term_coeff))) {
        all_matched = false;
        break;
      }
      matched.emplace_back(base_factor, needed);
    }
    if (!all_matched) {
      continue;
    }
    for (const auto& [base_factor, needed] : matched) {
      const Expr** term = coeff_of(base_factor);
      *term = a_.sub(*term, needed);
    }
    *coeff_of(factor) = zero();
    // _rewrite_mod_subtraction
    const Expr* floordiv = a_.function(Kind::FloorDiv, {base, divisor});
    replacements.push_back(
        a_.mul({a_.mul({a_.neg(mod_coeff), floordiv}), divisor}));
  }
  if (replacements.empty()) {
    return e;
  }
  c10::SmallVector<const Expr*, 8> new_terms;
  for (const auto& [factor, coeff] : terms) {
    if (coeff == zero()) {
      continue;
    }
    if (factor == one()) {
      new_terms.push_back(coeff);
    } else if (coeff == one()) {
      new_terms.push_back(factor);
    } else {
      new_terms.push_back(a_.mul({coeff, factor}));
    }
  }
  new_terms.append(replacements.begin(), replacements.end());
  return a_.add(new_terms);
}

const Expr* Analysis::rewrite(const Expr* e) {
  if (e->args.empty()) {
    return e;
  }
  c10::SmallVector<const Expr*, 8> args;
  bool changed = false;
  for (const Expr* arg : e->args) {
    args.push_back(rewrite(arg));
    changed |= args.back() != arg;
  }
  if (changed) {
    // expr.func(*args)
    if (e->is_relational()) {
      e = a_.rel(e->kind, args[0], args[1]);
    } else {
      switch (e->kind) {
        case Kind::Add:
          e = a_.add(args);
          break;
        case Kind::Mul:
          e = a_.mul(args);
          break;
        case Kind::Pow:
          e = a_.pow(args[0], args[1]);
          break;
        case Kind::Not:
          e = a_.logical_not(args[0]);
          break;
        case Kind::And:
          e = a_.logical_and(args);
          break;
        case Kind::Or:
          e = a_.logical_or(args);
          break;
        default:
          e = a_.function(e->kind, args);
      }
    }
  }
  return e->kind == Kind::Add ? rewrite_mod_subtractions_in_add(e) : e;
}

bool has_mod(const Expr* e) {
  if (e->kind == Kind::Mod || e->kind == Kind::PythonMod) {
    return true;
  }
  return std::any_of(e->args.begin(), e->args.end(), has_mod);
}

} // namespace

ValueRanges::ValueRanges(const Expr* lower, const Expr* upper)
    : lower(lower), upper(upper) {
  if (lower->is_boolean() != upper->is_boolean()) {
    throw NativeUnsupported("mixed bool and numeric bounds");
  }
  if (lower->is_boolean()) {
    bool atoms = (lower->kind == Kind::BooleanTrue ||
                  lower->kind == Kind::BooleanFalse) &&
        (upper->kind == Kind::BooleanTrue || upper->kind == Kind::BooleanFalse);
    if (!atoms) {
      throw NativeUnsupported("not simple sympy type");
    }
    if (lower->kind == Kind::BooleanTrue && upper->kind == Kind::BooleanFalse) {
      throw NativeUnsupported("Invalid ranges");
    }
    return;
  }
  if (!lower->is_number() || !upper->is_number()) {
    throw NativeUnsupported("symbolic value range bound");
  }
  if (lt(upper, lower)) {
    throw NativeUnsupported("Invalid ranges");
  }
  if ((lower->kind == Kind::Integer && upper->kind == Kind::Infinity) ||
      (upper->kind == Kind::Integer && lower->kind == Kind::NegativeInfinity)) {
    throw NativeUnsupported("unnormalized value range");
  }
}

ValueRanges make_value_range(
    ExprArena& arena,
    const Expr* lower,
    const Expr* upper) {
  if (lower->kind == Kind::Integer && upper == arena.oo()) {
    upper = arena.int_oo();
  }
  if (upper->kind == Kind::Integer && lower == arena.neg_oo()) {
    lower = arena.neg_int_oo();
  }
  return {lower, upper};
}

ValueRanges value_range_interp(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges) {
  return Analysis(arena, ranges).interp(e);
}

ValueRanges bound_sympy(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges,
    const RangeMap* context_ranges) {
  if (e->is_number()) {
    return {e, e};
  }
  Analysis analysis(arena, ranges, context_ranges);
  if (has_mod(e)) {
    e = analysis.rewrite(e);
  }
  return analysis.interp(e);
}

} // namespace torch::symbolic
