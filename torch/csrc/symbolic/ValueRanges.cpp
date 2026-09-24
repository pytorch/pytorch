#include <torch/csrc/symbolic/ValueRanges.h>

#include <c10/util/SmallVector.h>

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

bool is_int_bound(const Expr* e) {
  return e->kind == Kind::Integer || is_int_oo(e);
}

int sign(const Expr* e) {
  if (is_int_oo(e)) {
    return e->kind == Kind::IntInfinity ? 1 : -1;
  }
  return (e->p > 0) - (e->p < 0);
}

bool lt(const Expr* a, const Expr* b) {
  return compare_numbers(a, b) < 0;
}

// Python's min(a, b) and max(a, b).
const Expr* min_num(const Expr* a, const Expr* b) {
  return lt(b, a) ? b : a;
}

const Expr* max_num(const Expr* a, const Expr* b) {
  return lt(a, b) ? b : a;
}

void require_int(const ValueRanges& r) {
  if (r.is_bool()) {
    throw NativeUnsupported("bool range in an integer handler");
  }
}

void require_bool(const ValueRanges& r) {
  if (!r.is_bool()) {
    throw NativeUnsupported("not bool like");
  }
}

class Analysis {
 public:
  Analysis(ExprArena& arena, const RangeMap& ranges)
      : a_(arena), ranges_(ranges) {}

  ValueRanges interp(const Expr* e);

 private:
  const Expr* zero() {
    return a_.integer(0);
  }
  const Expr* one() {
    return a_.integer(1);
  }
  ValueRanges unknown_int() {
    return {a_.neg_int_oo(), a_.int_oo()};
  }
  ValueRanges unknown_bool() {
    return {a_.boolean(false), a_.boolean(true)};
  }
  bool contains_zero(const ValueRanges& r) {
    require_int(r);
    return !lt(zero(), r.lower) && !lt(r.upper, zero());
  }

  const Expr* num_add(const Expr* x, const Expr* y);
  const Expr* num_sub(const Expr* x, const Expr* y) {
    return num_add(x, a_.neg(y));
  }
  const Expr* num_abs(const Expr* x) {
    return is_int_oo(x) ? a_.int_oo() : x->p < 0 ? a_.neg(x) : x;
  }
  const Expr* safe_mul(const Expr* x, const Expr* y);
  const Expr* safe_pow(const Expr* base, const Expr* exp);
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
  ValueRanges increasing_map(Kind kind, const ValueRanges& x);

  ExprArena& a_;
  const RangeMap& ranges_;
};

const Expr* Analysis::num_add(const Expr* x, const Expr* y) {
  if (is_int_oo(x) || is_int_oo(y)) {
    if (is_int_oo(x) && is_int_oo(y) && x != y) {
      throw NativeUnsupported("sympy expression is NaN");
    }
    return is_int_oo(x) ? x : y;
  }
  return a_.add({x, y});
}

// safe_mul in SymPyValueRangeAnalysis.mul.
const Expr* Analysis::safe_mul(const Expr* x, const Expr* y) {
  if (sign(x) == 0) {
    return x;
  }
  if (sign(y) == 0) {
    return y;
  }
  if (is_int_oo(x) || is_int_oo(y)) {
    return sign(x) * sign(y) > 0 ? a_.int_oo() : a_.neg_int_oo();
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
  return a_.function(Kind::PowByNatural, {base, exp});
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
    throw NativeUnsupported("float value range");
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
    throw NativeUnsupported("lt of a bool and an integer range");
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
  require_int(x);
  require_int(y);
  return {num_add(x.lower, y.lower), num_add(x.upper, y.upper)};
}

ValueRanges Analysis::mul(const ValueRanges& x, const ValueRanges& y) {
  if (x.is_bool() != y.is_bool()) {
    throw NativeUnsupported("mul of a bool and an integer range");
  }
  if (x.is_bool()) {
    return and_(x, y);
  }
  // coordinatewise_monotone_map
  const Expr* products[] = {
      safe_mul(x.lower, y.lower),
      safe_mul(x.lower, y.upper),
      safe_mul(x.upper, y.lower),
      safe_mul(x.upper, y.upper)};
  const Expr* lo = products[0];
  const Expr* hi = products[0];
  for (const Expr* p : products) {
    lo = min_num(lo, p);
    hi = max_num(hi, p);
  }
  return {lo, hi};
}

ValueRanges Analysis::floordiv(const ValueRanges& x, const ValueRanges& y) {
  require_int(x);
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
  const Expr* lo = nullptr;
  const Expr* hi = nullptr;
  for (const Expr* xb : {x.lower, x.upper}) {
    for (const Expr* yb : {y.lower, y.upper}) {
      // FloorDiv is nan when both sides are infinite.
      const Expr* r = is_int_oo(xb) && is_int_oo(yb)
          ? (sign(xb) * sign(yb) > 0 ? a_.int_oo() : a_.neg_int_oo())
          : a_.function(Kind::FloorDiv, {xb, yb});
      lo = lo == nullptr ? r : min_num(lo, r);
      hi = hi == nullptr ? r : max_num(hi, r);
    }
  }
  return {lo, hi};
}

// C semantics, like SymPyValueRangeAnalysis.mod.
ValueRanges Analysis::mod(const ValueRanges& x, const ValueRanges& y) {
  require_int(x);
  if (contains_zero(y)) {
    return unknown_int();
  }
  if (y.is_singleton()) {
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
  return {a_.neg(upper), upper};
}

ValueRanges Analysis::python_mod(const ValueRanges& x, const ValueRanges& y) {
  require_int(x);
  require_int(y);
  if (!lt(x.lower, zero()) && !lt(y.lower, zero())) {
    return mod(x, y);
  }
  return {
      lt(y.lower, zero()) ? num_add(y.lower, one()) : zero(),
      lt(zero(), y.upper) ? num_sub(y.upper, one()) : zero()};
}

ValueRanges Analysis::pow_by_natural(
    const ValueRanges& x,
    const ValueRanges& y) {
  require_int(x);
  require_int(y);
  if (x.is_singleton() && y.is_singleton()) {
    const Expr* r = safe_pow(x.lower, y.lower);
    return {r, r};
  }
  if (!lt(x.lower, one())) {
    // y & ValueRanges(0, int_oo)
    ValueRanges exp(max_num(y.lower, zero()), y.upper);
    return {
        pow_by_natural_fn(x.lower, exp.lower),
        pow_by_natural_fn(x.upper, exp.upper)};
  }
  if (y.is_singleton()) {
    const Expr* l = safe_pow(x.lower, y.lower);
    const Expr* u = safe_pow(x.upper, y.lower);
    if (y.lower->p % 2 == 1) {
      return {l, u};
    }
    // convex_min_zero_map
    if (contains_zero(x)) {
      return {zero(), max_num(l, u)};
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
    throw NativeUnsupported("min/max of a bool and an integer range");
  }
  if (x.is_bool()) {
    return kind == Kind::Min ? and_(x, y) : or_(x, y);
  }
  auto fn = kind == Kind::Min ? min_num : max_num;
  return {fn(x.lower, y.lower), fn(x.upper, y.upper)};
}

ValueRanges Analysis::increasing_map(Kind kind, const ValueRanges& x) {
  require_int(x);
  return {a_.function(kind, {x.lower}), a_.function(kind, {x.upper})};
}

ValueRanges Analysis::interp(const Expr* e) {
  switch (e->kind) {
    case Kind::Integer:
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
    case Kind::BooleanTrue:
    case Kind::BooleanFalse:
      return {e, e};
    case Kind::Rational:
      throw NativeUnsupported("float value range");
    case Kind::Symbol: {
      auto it = ranges_.find(e);
      return it != ranges_.end() ? it->second : default_symbol_range(e);
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
      // Integer exponents only; a negative one takes the float pow handler.
      if (e->args[1]->kind != Kind::Integer || e->args[1]->p < 0) {
        throw NativeUnsupported("float value range");
      }
      return pow_by_natural(args[0], args[1]);
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
      // sympy.ceiling / sympy.floor are the identity on integer bounds.
      require_int(args[0]);
      return args[0];
    case Kind::TruncToInt:
    case Kind::RoundToInt:
      return increasing_map(e->kind, args[0]);
    case Kind::IsNonOverlappingAndDenseIndicator:
      return unknown_int();
    case Kind::FloatPow:
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
    case Kind::RoundDecimal:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
      throw NativeUnsupported("float value range");
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

} // namespace

ValueRanges::ValueRanges(const Expr* lower, const Expr* upper)
    : lower(lower), upper(upper) {
  if (lower->is_boolean() != upper->is_boolean()) {
    throw NativeUnsupported("mixed bool and integer bounds");
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
  if (!is_int_bound(lower) || !is_int_bound(upper)) {
    throw NativeUnsupported("float value range");
  }
  if (lt(upper, lower)) {
    throw NativeUnsupported("Invalid ranges");
  }
}

ValueRanges value_range_interp(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges) {
  return Analysis(arena, ranges).interp(e);
}

} // namespace torch::symbolic
