#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <limits>
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

} // namespace

const char* function_name(Kind k) {
  switch (k) {
    case Kind::Mod:
      return "Mod";
    case Kind::PythonMod:
      return "PythonMod";
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

} // namespace torch::symbolic
