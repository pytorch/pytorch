#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>

// Ports of the classes in torch/utils/_sympy/functions.py.

namespace torch::symbolic {

namespace {

using i128 = __int128;

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
    default:
      throw NativeUnsupported("not a function kind");
  }
}

const Expr* ExprArena::function(Kind kind, c10::ArrayRef<const Expr*> args) {
  if (args.size() != 2) {
    throw NativeUnsupported(
        std::string(function_name(kind)) + " takes exactly 2 arguments");
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

} // namespace torch::symbolic
