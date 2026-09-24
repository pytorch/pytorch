#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

// Ports of the classes in torch/utils/_sympy/functions.py.

namespace torch::symbolic {

namespace {

using i128 = __int128;

bool is_minmax(const Expr* e) {
  return e != nullptr && (e->kind == Kind::Max || e->kind == Kind::Min);
}

bool contains(c10::ArrayRef<const Expr*> xs, const Expr* x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

bool is_int_oo(const Expr* e) {
  return e->kind == Kind::IntInfinity || e->kind == Kind::NegativeIntInfinity;
}

i128 floordiv128(i128 a, i128 b) {
  i128 q = a / b;
  return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

int64_t to_int64(i128 v) {
  if (v < std::numeric_limits<int64_t>::min() ||
      v > std::numeric_limits<int64_t>::max()) {
    throw NativeUnsupported("integer overflow");
  }
  return static_cast<int64_t>(v);
}

// Add.make_args / Mul.make_args; `e` must outlive the result.
c10::ArrayRef<const Expr*> make_args(Kind kind, const Expr*& e) {
  return e->kind == kind ? c10::ArrayRef<const Expr*>(e->args)
                         : c10::ArrayRef<const Expr*>(e);
}

bool has_rational_atom(const Expr* e) {
  return e->kind == Kind::Rational ||
      std::any_of(e->args.begin(), e->args.end(), has_rational_atom);
}

// Integers, symbols and products of an Integer and powers of symbols with
// positive exponents: the expressions that sympy.simplify returns unchanged.
bool is_monomial(const Expr* e) {
  auto is_factor = [](const Expr* f) {
    return f->kind == Kind::Symbol ||
        (f->kind == Kind::Pow && f->args[0]->kind == Kind::Symbol &&
         f->args[1]->kind == Kind::Integer && f->args[1]->p > 0);
  };
  if (e->kind == Kind::Integer) {
    return true;
  }
  if (e->kind != Kind::Mul) {
    return is_factor(e);
  }
  return std::all_of(e->args.begin(), e->args.end(), [&](const Expr* f) {
    return is_factor(f) || (f == e->args[0] && f->kind == Kind::Integer);
  });
}

// sympy.gcd(p, q) == 1 for monomials p and an Add q of monomials when
// simple_floordiv_gcd(p, q) is 1 (so the integer contents are coprime) and
// every symbol of p is missing from some term of q.
bool monomial_coprime_to_sum(const Expr* p, const Expr* q) {
  auto symbol_of = [](const Expr* f) {
    return f->kind == Kind::Pow ? f->args[0] : f;
  };
  if (!is_monomial(p) ||
      !std::all_of(q->args.begin(), q->args.end(), is_monomial)) {
    return false;
  }
  for (const Expr* x : make_args(Kind::Mul, p)) {
    if (x->kind == Kind::Integer) {
      continue;
    }
    if (std::all_of(q->args.begin(), q->args.end(), [&](const Expr* t) {
          auto factors = make_args(Kind::Mul, t);
          return std::any_of(
              factors.begin(), factors.end(), [&](const Expr* f) {
                return symbol_of(f) == symbol_of(x);
              });
        })) {
      return false;
    }
  }
  return true;
}

// integer_factor in simple_floordiv_gcd.
int64_t integer_factor(const Expr* e) {
  int64_t r = 0;
  for (const Expr* t : make_args(Kind::Add, e)) {
    int64_t c = 1;
    for (const Expr* a : make_args(Kind::Mul, t)) {
      if (a->kind == Kind::Integer &&
          (a->p == std::numeric_limits<int64_t>::min() ||
           __builtin_mul_overflow(c, std::abs(a->p), &c))) {
        throw NativeUnsupported("integer overflow");
      }
    }
    r = std::gcd(r, c);
  }
  return r;
}

// torch.utils._sympy.functions.simple_floordiv_gcd.
const Expr* simple_floordiv_gcd(ExprArena& A, const Expr* p, const Expr* q) {
  int64_t g = std::gcd(integer_factor(p), integer_factor(q));
  if (g != 1) {
    const Expr* inv = A.pow(A.integer(g), A.integer(-1));
    p = A.mul({p, inv});
    q = A.mul({q, inv});
  }
  const Expr* gcd = A.integer(g);
  for (const Expr* x : make_args(Kind::Mul, q)) {
    auto terms = make_args(Kind::Add, p);
    if (std::all_of(terms.begin(), terms.end(), [&](const Expr* t) {
          auto split = make_args(Kind::Mul, t);
          return std::find(split.begin(), split.end(), x) != split.end();
        })) {
      gcd = A.mul({gcd, x});
    }
  }
  return gcd;
}

} // namespace

const char* function_name(Kind k) {
  switch (k) {
    case Kind::Mod:
      return "Mod";
    case Kind::PythonMod:
      return "PythonMod";
    case Kind::FloorDiv:
      return "FloorDiv";
    case Kind::CleanDiv:
      return "CleanDiv";
    case Kind::Max:
      return "Max";
    case Kind::Min:
      return "Min";
    case Kind::PowByNatural:
      return "PowByNatural";
    case Kind::FloatPow:
      return "FloatPow";
    case Kind::FloatTrueDiv:
      return "FloatTrueDiv";
    case Kind::IntTrueDiv:
      return "IntTrueDiv";
    case Kind::CeilToInt:
      return "CeilToInt";
    case Kind::FloorToInt:
      return "FloorToInt";
    case Kind::TruncToInt:
      return "TruncToInt";
    case Kind::RoundToInt:
      return "RoundToInt";
    case Kind::RoundDecimal:
      return "RoundDecimal";
    case Kind::ToFloat:
      return "ToFloat";
    case Kind::TruncToFloat:
      return "TruncToFloat";
    case Kind::IsNonOverlappingAndDenseIndicator:
      return "IsNonOverlappingAndDenseIndicator";
    default:
      throw NativeUnsupported("not a function kind");
  }
}

const Expr* ExprArena::function(Kind kind, c10::ArrayRef<const Expr*> args) {
  if (kind == Kind::Max || kind == Kind::Min) {
    return minmax(kind, args);
  }
  size_t arity = 2;
  switch (kind) {
    case Kind::CeilToInt:
    case Kind::FloorToInt:
    case Kind::TruncToInt:
    case Kind::RoundToInt:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
      arity = 1;
      break;
    case Kind::IsNonOverlappingAndDenseIndicator:
      arity = args.size();
      break;
    default:
      break;
  }
  if (args.size() != arity) {
    throw NativeUnsupported(
        std::string(function_name(kind)) + " takes exactly " +
        std::to_string(arity) + " arguments");
  }
  for (const Expr* a : args) {
    if (a->is_boolean()) {
      throw NativeUnsupported("Boolean argument to a function");
    }
  }
  const Expr* r = nullptr;
  switch (kind) {
    case Kind::Mod:
    case Kind::PythonMod:
      r = eval_mod(kind, args[0], args[1]);
      break;
    case Kind::FloorDiv:
    case Kind::CleanDiv:
      r = eval_floordiv(args[0], args[1]);
      break;
    case Kind::PowByNatural:
      r = eval_pow_by_natural(args[0], args[1]);
      break;
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
      if (ask(args[1], Fact::zero) == Tri::True) {
        throw NativeUnsupported("division by zero");
      }
      [[fallthrough]];
    case Kind::FloatPow:
    case Kind::RoundDecimal:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
      // The folds of numbers return sympy.Floats or oo, or raise for
      // TruncToFloat(int_oo). The rest (IntTrueDiv of finite non-Integers,
      // RoundDecimal with a non-Integer ndigits, ToFloat of a Rational) stay
      // unevaluated, which the port rejects as well.
      if (std::all_of(args.begin(), args.end(), [](const Expr* a) {
            return a->is_number();
          })) {
        throw NativeUnsupported("function of numbers gives a Float");
      }
      break;
    case Kind::CeilToInt:
    case Kind::FloorToInt:
    case Kind::TruncToInt:
    case Kind::RoundToInt:
      r = eval_to_int(kind, args[0]);
      break;
    case Kind::IsNonOverlappingAndDenseIndicator:
      r = eval_is_non_overlapping_and_dense(args);
      break;
    default:
      throw NativeUnsupported("not a function kind");
  }
  if (r != nullptr) {
    return r;
  }
  if (std::all_of(args.begin(), args.end(), [](const Expr* a) {
        return a->is_number();
      })) {
    // sympy's is_number would hold for it, which the handlers rely on never
    // happening for non-Numbers.
    throw NativeUnsupported("unevaluated function of numbers");
  }
  return intern(kind, 0, 0, args);
}

const Expr* ExprArena::eval_mod(Kind kind, const Expr* p, const Expr* q) {
  if (ask(q, Fact::zero) == Tri::True) {
    throw NativeUnsupported("Modulo by zero");
  }
  if (p == zero_ || p == q || p == neg(q) || q == one_) {
    return zero_;
  }
  if (p->is_number() && q->is_number()) {
    // Number.__mod__; int_oo % x is nan.
    if (is_int_oo(p) || is_int_oo(q)) {
      throw NativeUnsupported("int_oo in Mod");
    }
    if (kind == Kind::Mod && (p->p < 0 || q->p < q->q)) {
      throw NativeUnsupported("AssertionError in Mod");
    }
    // Rational.__mod__, which is Python's % for Integers.
    i128 n = floordiv128(i128(p->p) * q->q, i128(q->p) * p->q);
    i128 num = i128(p->p) * q->q - n * q->p * p->q;
    return rational(to_int64(num), to_int64(i128(p->q) * q->q));
  }
  if (q->kind == Kind::Integer && q->p == 2) {
    if (ask(p, Fact::even) == Tri::True) {
      return zero_;
    }
    if (ask(p, Fact::odd) == Tri::True) {
      return one_;
    }
  }
  if (is_int_oo(p) || is_int_oo(q)) {
    throw NativeUnsupported("int_oo in Mod");
  }
  const Expr* r = mul({p, pow(q, neg_one_)});
  if (ask(r, Fact::integer) == Tri::True) {
    return zero_;
  }
  if (rel(Kind::Lt, p, q) == true_ && ask(r, Fact::positive) == Tri::True) {
    return p;
  }
  if (kind == Kind::PythonMod) {
    // `sympy.Mod(p, q) == 0` needs sympy's Mod.eval, including polynomial gcd.
    throw NativeUnsupported("PythonMod needs sympy.Mod");
  }
  return nullptr;
}

const Expr* ExprArena::eval_floordiv(const Expr* base, const Expr* divisor) {
  if (ask(divisor, Fact::zero) == Tri::True) {
    throw NativeUnsupported("division by zero");
  }
  if (is_int_oo(base) && is_int_oo(divisor)) {
    throw NativeUnsupported("FloorDiv gives nan");
  }
  if (ask(base, Fact::zero) == Tri::True) {
    return zero_;
  }
  if (ask(base, Fact::integer) == Tri::True && divisor == one_) {
    return base;
  }
  if (ask(base, Fact::integer) == Tri::True && divisor == neg_one_) {
    return mul({base, neg_one_});
  }
  if (base == divisor) {
    return one_;
  }
  if (base->is_number() && divisor->is_number() &&
      (is_int_oo(base) || is_int_oo(divisor))) {
    // floor(float(base) / float(divisor)), where exactly one side is infinite
    // and the divisor is nonzero; a finite divisor here is a Rational.
    if (is_int_oo(divisor)) {
      return zero_;
    }
    return (base == neg_int_oo_) != (divisor->p < 0) ? neg_int_oo_ : int_oo_;
  }
  if (base->kind == Kind::Integer && divisor->kind == Kind::Integer) {
    return integer(to_int64(floordiv128(base->p, divisor->p)));
  }
  if (base->kind == Kind::FloorDiv || base->kind == Kind::CleanDiv) {
    return function(
        Kind::FloorDiv, {base->args[0], mul({base->args[1], divisor})});
  }
  if (is_int_oo(base) || is_int_oo(divisor)) {
    throw NativeUnsupported("int_oo in FloorDiv");
  }

  if (divisor->kind == Kind::Integer) {
    const Expr* inv = pow(divisor, neg_one_);
    const Expr* quotients = zero_;
    c10::SmallVector<const Expr*, 4> rest;
    bool found = false;
    for (const Expr* term : make_args(Kind::Add, base)) {
      const Expr* quotient = mul({term, inv});
      // The sympy < 1.15 branch: sympy can report a Mul with a Rational
      // coefficient as an integer.
      if (ask(quotient, Fact::integer) == Tri::True &&
          (quotient->kind != Kind::Mul || !has_rational_atom(quotient))) {
        found = true;
        quotients = add({quotients, quotient});
      } else {
        rest.push_back(term);
      }
    }
    if (found) {
      // base - Add(*terms, evaluate=False) is the Add of the other terms.
      const Expr* r = rest.empty() ? zero_ : add(rest);
      return add({function(Kind::FloorDiv, {r, divisor}), quotients});
    }
  }

  const Expr* gcd = simple_floordiv_gcd(*this, base, divisor);
  if (gcd == one_ && divisor->kind == Kind::Add) {
    auto is_wide = [](const Expr* e) {
      return e->kind == Kind::Add && e->args.size() > 20;
    };
    // For a wide Add, safe_gcd falls back to simple_floordiv_gcd, which
    // gives 1 again.
    if (!is_wide(base) && !is_wide(divisor) &&
        !monomial_coprime_to_sum(base, divisor)) {
      throw NativeUnsupported("FloorDiv needs sympy.gcd");
    }
  }
  if (gcd != one_) {
    const Expr* inv = pow(gcd, neg_one_);
    const Expr* b = mul({base, inv});
    const Expr* d = mul({divisor, inv});
    if (!is_monomial(b) || !is_monomial(d)) {
      throw NativeUnsupported("FloorDiv needs sympy.simplify");
    }
    return function(Kind::FloorDiv, {b, d});
  }
  return nullptr;
}

const Expr* ExprArena::eval_pow_by_natural(const Expr* base, const Expr* exp) {
  if (base->kind == Kind::Integer && exp->kind == Kind::Integer) {
    if (exp->p < 0) {
      throw NativeUnsupported("Exponent must be non-negative.");
    }
    // safe_pow; std::nullopt is int_oo.
    constexpr i128 kMaxSize = std::numeric_limits<int64_t>::max();
    auto safe_pow = [&](auto& self, i128 b, int64_t e) -> std::optional<i128> {
      if (e == 0) {
        return 1;
      }
      std::optional<i128> half = self(self, b, e / 2);
      if (!half || *half * *half > kMaxSize) {
        return std::nullopt;
      }
      i128 r = *half * *half;
      if (e % 2 == 1) {
        r *= b;
        if (r > kMaxSize) {
          return std::nullopt;
        }
      }
      return r;
    };
    bool negative = base->p < 0 && exp->p % 2 == 1;
    std::optional<i128> r =
        safe_pow(safe_pow, base->p < 0 ? -i128(base->p) : base->p, exp->p);
    if (!r) {
      return negative ? neg_int_oo_ : int_oo_;
    }
    return integer(static_cast<int64_t>(negative ? -*r : *r));
  }
  if (exp->kind == Kind::Integer) {
    return pow(base, exp);
  }
  if (exp == int_oo_) {
    if (ask(base, Fact::nonnegative) == Tri::True) {
      return int_oo_;
    }
    if (ask(base, Fact::negative) == Tri::True) {
      throw NativeUnsupported("PowByNatural gives zoo");
    }
  }
  return nullptr;
}

const Expr* ExprArena::eval_to_int(Kind kind, const Expr* number) {
  if (is_int_oo(number)) {
    if (kind == Kind::RoundToInt) {
      // RoundToInt only checks for the float oo; float(int_oo) is inf.
      throw NativeUnsupported("cannot convert float infinity to integer");
    }
    return number;
  }
  if (kind == Kind::TruncToInt) {
    if (ask(number, Fact::integer) == Tri::True) {
      return number;
    }
    if (number->kind == Kind::IntTrueDiv) {
      if (number->args[1] == one_) {
        return number->args[0];
      }
      if (number->args[1] == neg_one_) {
        return neg(number->args[0]);
      }
    }
  }
  if (kind == Kind::FloorToInt && number->kind == Kind::Integer) {
    return number;
  }
  if (!number->is_rational()) {
    return nullptr;
  }
  // float(number), correctly rounded like sympy's Rational.__float__.
  constexpr int64_t kExact = int64_t(1) << 53;
  if (number->kind == Kind::Rational &&
      (number->q > kExact || number->p > kExact || number->p < -kExact)) {
    throw NativeUnsupported("float of a large Rational");
  }
  double x = static_cast<double>(number->p) / static_cast<double>(number->q);
  double r = 0;
  switch (kind) {
    case Kind::CeilToInt:
      r = std::ceil(x);
      break;
    case Kind::FloorToInt:
      r = std::floor(x);
      break;
    case Kind::TruncToInt:
      r = std::trunc(x);
      break;
    default:
      // round(x, 0): to nearest, halfway cases to even.
      r = std::round(x);
      if (std::fabs(x - r) == 0.5) {
        r = 2.0 * std::round(x / 2.0);
      }
      break;
  }
  if (!(r >= -0x1p63 && r < 0x1p63)) {
    throw NativeUnsupported("integer overflow");
  }
  return integer(static_cast<int64_t>(r));
}

const Expr* ExprArena::eval_is_non_overlapping_and_dense(
    c10::ArrayRef<const Expr*> args) {
  if (args.size() % 2 != 0) {
    throw NativeUnsupported("expected an even number of arguments");
  }
  const size_t dim = args.size() / 2;
  auto sizes = args.slice(0, dim);
  auto strides = args.slice(dim);
  auto is_integer = [](const Expr* a) { return a->kind == Kind::Integer; };
  // eval_is_non_overlapping_and_dense over (size, stride) pairs.
  auto dense = [&](std::vector<std::pair<int64_t, int64_t>> dims) {
    if (dims.size() == 1) {
      return integer(dims[0].second == 1 || dims[0].first < 2);
    }
    std::stable_sort(dims.begin(), dims.end(), [](auto& a, auto& b) {
      return a.second < b.second;
    });
    i128 expected_stride = 1;
    for (auto [length, stride] : dims) {
      if (length == 1) {
        continue;
      }
      if (stride != expected_stride) {
        return zero_;
      }
      expected_stride *= length;
    }
    return one_;
  };
  std::vector<std::pair<int64_t, int64_t>> dims;
  if (std::all_of(args.begin(), args.end(), is_integer)) {
    for (size_t i = 0; i < dim; ++i) {
      dims.emplace_back(sizes[i]->p, strides[i]->p);
    }
    return dense(std::move(dims));
  }
  if (dim == 1) {
    if (strides[0] == one_) {
      return one_;
    }
    if (sizes[0]->is_number() && compare_numbers(sizes[0], integer(2)) < 0) {
      return one_;
    }
  }
  if (std::all_of(strides.begin(), strides.end(), is_integer)) {
    std::vector<size_t> order(dim);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return strides[a]->p < strides[b]->p;
    });
    // The size of the largest stride is ignored, so it may be symbolic.
    for (size_t i = 0; i < dim; ++i) {
      const Expr* size = sizes[order[i]];
      if (i + 1 < dim && !is_integer(size)) {
        return nullptr;
      }
      dims.emplace_back(i + 1 < dim ? size->p : 42, strides[order[i]]->p);
    }
    return dense(std::move(dims));
  }
  return nullptr;
}

const Expr* ExprArena::lshift(const Expr* base, const Expr* shift) {
  if (base->is_boolean() || shift->is_boolean()) {
    throw NativeUnsupported("Boolean argument to a function");
  }
  if (ask(shift, Fact::negative) == Tri::True) {
    throw NativeUnsupported("negative shift count");
  }
  const Expr* power = function(Kind::PowByNatural, {integer(2), shift});
  return mul({base, power});
}

const Expr* ExprArena::rshift(const Expr* base, const Expr* shift) {
  if (base->is_boolean() || shift->is_boolean()) {
    throw NativeUnsupported("Boolean argument to a function");
  }
  if (ask(shift, Fact::negative) == Tri::True) {
    throw NativeUnsupported("negative shift count");
  }
  const Expr* power = function(Kind::PowByNatural, {integer(2), shift});
  return function(Kind::FloorDiv, {base, power});
}

const Expr* ExprArena::ceildiv(const Expr* base, const Expr* divisor) {
  // CeilDiv.__new__. sympy.gcd(base, divisor) == divisor is decided for
  // polynomials with Integer coefficients and a monomial divisor, where it
  // holds iff the divisor's coefficient is positive (sympy.gcd normalizes the
  // sign) and the divisor divides every term of base. The exception 0/0
  // raises either way.
  auto terms = make_args(Kind::Add, base);
  if (!std::all_of(terms.begin(), terms.end(), is_monomial)) {
    throw NativeUnsupported("CeilDiv needs sympy.gcd");
  }
  auto exponent = [](const Expr* term, const Expr* x) -> int64_t {
    for (const Expr* f : make_args(Kind::Mul, term)) {
      if (f == x) {
        return 1;
      }
      if (f->kind == Kind::Pow && f->args[0] == x) {
        return f->args[1]->p;
      }
    }
    return 0;
  };
  bool is_gcd = false;
  if (is_monomial(divisor)) {
    int64_t c = as_coeff_Mul(divisor).first->p;
    auto factors = make_args(Kind::Mul, divisor);
    is_gcd = c > 0 &&
        (base == zero_ ||
         std::all_of(terms.begin(), terms.end(), [&](const Expr* t) {
           return as_coeff_Mul(t).first->p % c == 0 &&
               std::all_of(factors.begin(), factors.end(), [&](auto f) {
                    const Expr* x = f->kind == Kind::Pow ? f->args[0] : f;
                    return f->kind == Kind::Integer ||
                        exponent(t, x) >= exponent(f, x);
                  });
         }));
  } else if (
      divisor->kind != Kind::Add ||
      !std::all_of(divisor->args.begin(), divisor->args.end(), is_monomial) ||
      base == zero_ || base->kind == Kind::Add) {
    // A nonzero monomial has only monomial divisors.
    throw NativeUnsupported("CeilDiv needs sympy.gcd");
  }
  if (is_gcd) {
    return function(Kind::CleanDiv, {base, divisor});
  }
  return function(
      Kind::FloorDiv, {add({base, add({divisor, neg_one_})}), divisor});
}

const Expr* ExprArena::minmax(
    Kind kind,
    c10::ArrayRef<const Expr*> args,
    bool evaluate) {
  // MinMaxBase.__new__. cls.zero and cls.identity are the float oo and -oo,
  // which are not in the arena; nullptr stands for cls.identity where
  // _collapse_arguments inserts it. The unique_summations_symbols fast path
  // only skips _collapse_arguments and _find_localzeros where they return
  // their input unchanged, so it is not ported.
  const bool is_max = kind == Kind::Max;
  const Kind other = is_max ? Kind::Min : Kind::Max;
  auto build = [&](Kind k, c10::ArrayRef<const Expr*> xs) {
    std::vector<const Expr*> sorted = ordered_frozenset(xs);
    if (sorted.empty()) {
      throw NativeUnsupported("Max/Min of no arguments");
    }
    if (sorted.size() == 1) {
      return sorted[0];
    }
    if (std::all_of(sorted.begin(), sorted.end(), [](const Expr* a) {
          return a->is_number();
        })) {
      throw NativeUnsupported("unevaluated function of numbers");
    }
    return intern(k, 0, 0, sorted);
  };
  for (const Expr* a : args) {
    if (a->is_boolean()) {
      throw NativeUnsupported("Boolean argument to Max/Min");
    }
  }
  if (!evaluate) {
    return build(kind, args);
  }

  // _new_args_filter.
  std::vector<const Expr*> flat;
  for (const Expr* a : args) {
    if (ask(a, Fact::extended_real) == Tri::False) {
      throw NativeUnsupported("Max/Min argument is not comparable");
    }
    if (a->kind == kind) {
      flat.insert(flat.end(), a->args.begin(), a->args.end());
    } else {
      flat.push_back(a);
    }
  }

  // _collapse_arguments.
  std::vector<const Expr*> xs = ordered_frozenset(flat);
  if (xs.empty()) {
    throw NativeUnsupported("Max/Min of no arguments");
  }
  if (xs[0]->is_number()) {
    // nullptr is Min.identity for small and Max.identity for big.
    const Expr* small = nullptr;
    const Expr* big = nullptr;
    auto walk = [&](auto& self, const Expr* v) -> void {
      if (!is_minmax(v)) {
        return;
      }
      const Expr* v0 = v->args[0];
      if (v0->is_number()) {
        const Expr*& t = v->kind == Kind::Min ? small : big;
        int sign = v->kind == Kind::Min ? -1 : 1;
        if (t == nullptr || compare_numbers(v0, t) == sign) {
          t = v0;
        }
      }
      for (const Expr* a : v->args) {
        self(self, a);
      }
    };
    for (const Expr* x : xs) {
      walk(walk, x);
    }
    const Expr*& t = is_max ? big : small;
    for (const Expr* x : xs) {
      if (!x->is_number()) {
        break;
      }
      if (t == nullptr || compare_numbers(x, t) == (is_max ? 1 : -1)) {
        t = x;
      }
    }
    if (t != nullptr) {
      for (const Expr*& x : xs) {
        if (x->kind != other) {
          continue;
        }
        // (a0 > T) == True or (a0 < T) == True; int_oo's operator overloads
        // agree with the numeric order.
        const Expr* a0 = x->args[0];
        Kind op = other == Kind::Max ? Kind::Gt : Kind::Lt;
        bool redundant = a0->is_number()
            ? compare_numbers(a0, t) == (op == Kind::Gt ? 1 : -1)
            : rel(op, a0, t) == true_;
        if (redundant) {
          x = nullptr;
        }
      }
    }
  }

  auto do_ = [&](auto& self, const Expr* ai, const Expr* a) -> const Expr* {
    if (!is_minmax(ai)) {
      return ai;
    }
    std::vector<const Expr*> sub;
    bool cond = contains(ai->args, a);
    if (cond && ai->kind != kind) {
      return a;
    }
    for (const Expr* i : ai->args) {
      if (i != a) {
        sub.push_back(self(self, i, a));
      }
    }
    if (!cond && std::equal(sub.begin(), sub.end(), ai->args.begin())) {
      return ai;
    }
    return build(ai->kind, sub);
  };
  for (size_t i = 0; i < xs.size(); ++i) {
    // No Min/Max has cls.identity among its args.
    if (xs[i] == nullptr) {
      continue;
    }
    for (size_t j = i + 1; j < xs.size(); ++j) {
      xs[j] = do_(do_, xs[j], xs[i]);
    }
  }

  // factor_minmax.
  if (xs.size() > 1) {
    std::vector<const Expr*> others;
    std::vector<const Expr*> remaining;
    for (const Expr* x : xs) {
      (x != nullptr && x->kind == other ? others : remaining).push_back(x);
    }
    std::vector<const Expr*> common;
    if (!others.empty()) {
      for (const Expr* a : others[0]->args) {
        if (std::all_of(others.begin(), others.end(), [&](const Expr* o) {
              return contains(o->args, a);
            })) {
          common.push_back(a);
        }
      }
    }
    if (!common.empty()) {
      std::vector<std::vector<const Expr*>> diffs;
      for (const Expr* o : others) {
        auto& diff = diffs.emplace_back();
        for (const Expr* a : o->args) {
          if (!contains(common, a)) {
            diff.push_back(a);
          }
        }
      }
      std::vector<const Expr*> new_other_args = common;
      if (std::none_of(diffs.begin(), diffs.end(), [](const auto& d) {
            return d.empty();
          })) {
        std::vector<const Expr*> other_args_diff;
        for (const auto& d : diffs) {
          other_args_diff.push_back(build(other, d));
        }
        new_other_args.push_back(build(kind, other_args_diff));
      }
      remaining.push_back(build(other, new_other_args));
      xs = std::move(remaining);
    }
  }

  // _find_localzeros. cls.identity loses every comparison with a Number, and
  // there is always one when it was inserted.
  const Expr* num = nullptr;
  bool saw_identity = false;
  std::vector<const Expr*> values;
  for (const Expr* x : xs) {
    if (x == nullptr) {
      saw_identity = true;
    } else if (x->is_number()) {
      if (num == nullptr || compare_numbers(x, num) == (is_max ? 1 : -1)) {
        num = x;
      }
    } else if (!contains(values, x)) {
      values.push_back(x);
    }
  }
  if (num == nullptr) {
    if (saw_identity) {
      throw NativeUnsupported("Max/Min of cls.identity");
    }
    if (values.size() == 2) {
      // _collapse_known_multiplicative_terms; the result does not depend on
      // which of the two set elements is a.
      const Expr* a = values[0];
      const Expr* b = values[1];
      auto [ac, at] = as_coeff_Mul(a);
      auto [bc, bt] = as_coeff_Mul(b);
      if (at == bt) {
        std::optional<bool> a_smaller;
        if (ac == bc) {
          return a;
        } else if (ask(at, Fact::nonnegative) == Tri::True) {
          a_smaller = compare_numbers(ac, bc) < 0;
        } else if (ask(at, Fact::nonpositive) == Tri::True) {
          a_smaller = compare_numbers(ac, bc) > 0;
        }
        if (a_smaller.has_value()) {
          return *a_smaller != is_max ? a : b;
        }
      }
    }
  } else if (values.empty()) {
    return num;
  } else {
    if (values.size() == 1) {
      const Expr* o = values[0];
      if ((num == zero_ && ask(o, Fact::nonnegative) == Tri::True) ||
          (num == one_ && ask(o, Fact::positive) == Tri::True)) {
        return is_max ? o : num;
      }
    }
    values.push_back(num);
  }
  return build(kind, values);
}

} // namespace torch::symbolic
