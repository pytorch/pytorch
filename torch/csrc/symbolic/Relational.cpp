#include <torch/csrc/symbolic/Expr.h>

namespace torch::symbolic {

namespace {

using F = Fact;
using i128 = __int128;

Tri tri(bool b) {
  return b ? Tri::True : Tri::False;
}

Tri fuzzy_not(Tri t) {
  return t == Tri::Unknown ? t : tri(t == Tri::False);
}

Tri fuzzy_xor(Tri a, Tri b) {
  if (a == Tri::Unknown || b == Tri::Unknown) {
    return Tri::Unknown;
  }
  return tri(a != b);
}

bool is_boolean_atom(const Expr* e) {
  return e->kind == Kind::BooleanTrue || e->kind == Kind::BooleanFalse;
}

bool is_int_oo(const Expr* e) {
  return e->kind == Kind::IntInfinity || e->kind == Kind::NegativeIntInfinity;
}

// Position on the extended integer line, for comparing numbers where at least
// one is +-int_oo.
int infinity_rank(const Expr* e) {
  return e->kind == Kind::IntInfinity        ? 1
      : e->kind == Kind::NegativeIntInfinity ? -1
                                             : 0;
}

Kind reversed_kind(Kind k) {
  switch (k) {
    case Kind::Lt:
      return Kind::Gt;
    case Kind::Le:
      return Kind::Ge;
    case Kind::Gt:
      return Kind::Lt;
    case Kind::Ge:
      return Kind::Le;
    default:
      return k;
  }
}

void check_relational(const Expr* r) {
  if (!r->is_relational()) {
    throw NativeUnsupported("expected a Relational");
  }
}

} // namespace

Tri ExprArena::is_ge(const Expr* lhs, const Expr* rhs) {
  if (lhs->is_boolean() || rhs->is_boolean()) {
    // TypeError("Can only compare inequalities with Expr").
    throw NativeUnsupported("inequality of a Boolean");
  }
  // _eval_is_ge has no handlers for these kinds; then _n2.
  if (lhs->is_rational() && rhs->is_rational()) {
    return tri(i128(lhs->p) * rhs->q >= i128(rhs->p) * lhs->q);
  }
  if (lhs->is_number() && rhs->is_number()) {
    // int_oo - int_oo is nan, so neither _n2 nor the difference decides.
    if (lhs == rhs) {
      return Tri::Unknown;
    }
    return tri(infinity_rank(lhs) >= infinity_rank(rhs));
  }
  if (ask(lhs, F::extended_real) == Tri::True &&
      ask(rhs, F::extended_real) == Tri::True) {
    if ((ask(lhs, F::infinite) == Tri::True &&
         ask(lhs, F::extended_positive) == Tri::True) ||
        (ask(rhs, F::infinite) == Tri::True &&
         ask(rhs, F::extended_negative) == Tri::True)) {
      return Tri::True;
    }
    if (is_int_oo(lhs) || is_int_oo(rhs)) {
      throw NativeUnsupported("is_ge of int_oo and a symbolic expression");
    }
    return ask(sub(lhs, rhs), F::extended_nonnegative);
  }
  return Tri::Unknown;
}

Tri ExprArena::is_eq(const Expr* lhs, const Expr* rhs) {
  // No _eval_Eq or _eval_is_eq handler applies to these kinds.
  if (lhs == rhs) {
    return Tri::True;
  }
  if (is_boolean_atom(lhs) && is_boolean_atom(rhs)) {
    return Tri::False;
  }
  if (lhs->kind != Kind::Symbol && rhs->kind != Kind::Symbol &&
      lhs->is_boolean() != rhs->is_boolean()) {
    return Tri::False;
  }
  if (ask(lhs, F::infinite) == Tri::True ||
      ask(rhs, F::infinite) == Tri::True) {
    if (fuzzy_xor(ask(lhs, F::infinite), ask(rhs, F::infinite)) == Tri::True) {
      return Tri::False;
    }
    Tri lr = ask(lhs, F::extended_real);
    Tri rr = ask(rhs, F::extended_real);
    if (fuzzy_xor(lr, rr) == Tri::True) {
      return Tri::False;
    }
    if (lr == Tri::True && rr == Tri::True) {
      return fuzzy_xor(
          ask(lhs, F::extended_positive),
          fuzzy_not(ask(rhs, F::extended_positive)));
    }
    throw NativeUnsupported("is_eq of infinite non-real values");
  }
  if (lhs->is_boolean() || rhs->is_boolean()) {
    return Tri::Unknown;
  }
  if (lhs->is_number() && rhs->is_number()) {
    // Distinct numbers: int_oo minus any other number is +-int_oo.
    return Tri::False;
  }
  if (is_int_oo(lhs) || is_int_oo(rhs)) {
    throw NativeUnsupported("is_eq of int_oo and a symbolic expression");
  }
  const Expr* dif = sub(lhs, rhs);
  Tri z = ask(dif, F::zero);
  if (z == Tri::False && ask(dif, F::commutative) == Tri::True) {
    return Tri::False;
  }
  if (z == Tri::True) {
    return Tri::True;
  }
  // Without Floats the Float coefficient check cannot apply, and _n2 needs
  // both sides to be numbers, whose difference is_zero already decided.
  auto [n, d] = as_numer_denom(dif);
  if (ask(n, F::zero) == Tri::True) {
    return ask(d, F::nonzero);
  }
  if (ask(n, F::finite) == Tri::True) {
    if (ask(d, F::infinite) == Tri::True) {
      return Tri::True;
    }
    if (ask(n, F::zero) == Tri::False) {
      Tri rv = ask(d, F::infinite);
      if (rv == Tri::Unknown) {
        throw NativeUnsupported("is_eq needs clear_coefficients");
      }
      return rv;
    }
    return Tri::Unknown;
  }
  if (n->kind != Kind::Add) {
    return ask(n, F::infinite) == Tri::True ? Tri::False : Tri::Unknown;
  }
  for (const Expr* t : n->args) {
    if (ask(t, F::infinite) == Tri::True) {
      return Tri::False;
    }
  }
  return Tri::Unknown;
}

const Expr* ExprArena::rel(
    Kind kind,
    const Expr* lhs,
    const Expr* rhs,
    bool evaluate) {
  if (kind < Kind::Eq || kind > Kind::Ge) {
    throw NativeUnsupported("expected a Relational kind");
  }
  Tri v = Tri::Unknown;
  if (evaluate && (kind == Kind::Eq || kind == Kind::Ne)) {
    v = is_eq(lhs, rhs);
    v = kind == Kind::Eq ? v : fuzzy_not(v);
  } else if (evaluate) {
    // _Inequality.__new__ raises TypeError for these.
    for (const Expr* me : {lhs, rhs}) {
      if (ask(me, F::extended_real) == Tri::False) {
        throw NativeUnsupported("invalid comparison of non-real value");
      }
    }
    v = kind == Kind::Lt   ? fuzzy_not(is_ge(lhs, rhs))
        : kind == Kind::Le ? is_ge(rhs, lhs)
        : kind == Kind::Gt ? fuzzy_not(is_ge(rhs, lhs))
                           : is_ge(lhs, rhs);
  }
  if (v != Tri::Unknown) {
    return boolean(v == Tri::True);
  }
  return intern(kind, 0, 0, {lhs, rhs});
}

const Expr* ExprArena::logical_not(const Expr* a) {
  // Not.eval.
  if (a->is_rational()) {
    return boolean(a->p == 0);
  }
  if (is_int_oo(a) || a == true_) {
    return false_;
  }
  if (a == false_) {
    return true_;
  }
  if (a->kind == Kind::Not) {
    return a->args[0];
  }
  if (a->is_relational()) {
    return negated(a);
  }
  return intern(Kind::Not, 0, 0, {a});
}

const Expr* ExprArena::reversed(const Expr* r) {
  check_relational(r);
  return rel(reversed_kind(r->kind), r->args[1], r->args[0], false);
}

const Expr* ExprArena::reversedsign(const Expr* r) {
  check_relational(r);
  if (is_boolean_atom(r->args[0]) || is_boolean_atom(r->args[1])) {
    return r;
  }
  return rel(reversed_kind(r->kind), neg(r->args[0]), neg(r->args[1]), false);
}

const Expr* ExprArena::negated(const Expr* r) {
  check_relational(r);
  Kind k = r->kind;
  Kind n = k == Kind::Eq ? Kind::Ne
      : k == Kind::Ne    ? Kind::Eq
      : k == Kind::Ge    ? Kind::Lt
      : k == Kind::Gt    ? Kind::Le
      : k == Kind::Le    ? Kind::Gt
                         : Kind::Ge;
  return rel(n, r->args[0], r->args[1], false);
}

const Expr* ExprArena::weak(const Expr* r) {
  check_relational(r);
  if (r->kind == Kind::Lt || r->kind == Kind::Gt) {
    return rel(
        r->kind == Kind::Lt ? Kind::Le : Kind::Ge, r->args[0], r->args[1]);
  }
  return r;
}

const Expr* ExprArena::strict(const Expr* r) {
  check_relational(r);
  if (r->kind == Kind::Le || r->kind == Kind::Ge) {
    return rel(
        r->kind == Kind::Le ? Kind::Lt : Kind::Gt, r->args[0], r->args[1]);
  }
  return r;
}

} // namespace torch::symbolic
