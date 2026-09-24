#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <limits>
#include <numeric>

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

bool contains(c10::ArrayRef<const Expr*> xs, const Expr* x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

void check_relational(const Expr* r) {
  if (!r->is_relational()) {
    throw NativeUnsupported("expected a Relational");
  }
}

bool is_literal(const Expr* e) {
  // sympy.logic.boolalg.is_literal.
  if (e->kind == Kind::Not) {
    return is_literal(e->args[0]);
  }
  if (e->kind == Kind::And || e->kind == Kind::Or) {
    return false;
  }
  return std::all_of(e->args.begin(), e->args.end(), [](const Expr* a) {
    return a->args.empty();
  });
}

bool is_cnf(const Expr* e) {
  c10::ArrayRef<const Expr*> vals =
      e->kind == Kind::And ? c10::ArrayRef<const Expr*>(e->args) : e;
  for (const Expr* lit : vals) {
    c10::ArrayRef<const Expr*> vals2 =
        lit->kind == Kind::Or ? c10::ArrayRef<const Expr*>(lit->args) : lit;
    if (!std::all_of(vals2.begin(), vals2.end(), is_literal)) {
      return false;
    }
  }
  return true;
}

// is_nnf(e, simplified=False).
bool is_nnf(const Expr* e) {
  if (e->kind == Kind::And || e->kind == Kind::Or) {
    return std::all_of(e->args.begin(), e->args.end(), is_nnf);
  }
  return is_literal(e);
}

bool is_neg(ExprArena& arena, const Expr* t) {
  const Expr* c = t->kind == Kind::Mul ? t->args[0] : t;
  return c->is_number() && arena.ask(c, Fact::negative) == Tri::True;
}

int64_t integer_coefficient(const Expr* x) {
  const Expr* c = x->kind == Kind::Mul ? x->args[0] : x;
  if (c->kind != Kind::Integer) {
    return 1;
  }
  if (c->p == std::numeric_limits<int64_t>::min()) {
    throw NativeUnsupported("integer overflow");
  }
  return std::abs(c->p);
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

const Expr* ExprArena::as_boolean(const Expr* e) {
  // Integer(1) == True and Integer(0) == False.
  if (e->kind == Kind::Integer && (e->p == 0 || e->p == 1)) {
    return boolean(e->p == 1);
  }
  if (e->kind == Kind::Symbol) {
    Tri z = ask(e, F::zero);
    return z == Tri::Unknown ? e : boolean(z == Tri::False);
  }
  if (!e->is_boolean()) {
    throw NativeUnsupported("expecting bool or Boolean");
  }
  return e;
}

const Expr* ExprArena::lattice(Kind kind, c10::ArrayRef<const Expr*> args) {
  std::vector<const Expr*> sorted = ordered_frozenset(args);
  if (sorted.empty()) {
    return boolean(kind == Kind::And);
  }
  if (sorted.size() == 1) {
    return sorted[0];
  }
  return intern(kind, 0, 0, sorted);
}

const Expr* ExprArena::logical_and(c10::ArrayRef<const Expr*> args) {
  // And._new_args_filter: binary_check_and_simplify, the LatticeOp filter
  // (ordered() consumes all of it before any canonical), then relationals.
  c10::SmallVector<const Expr*, 8> checked;
  for (const Expr* a : args) {
    checked.push_back(as_boolean(a));
  }
  std::vector<const Expr*> flat;
  for (const Expr* a : checked) {
    if (a == false_) {
      return false_;
    }
    if (a == true_) {
      continue;
    }
    if (a->kind == Kind::And) {
      flat.insert(flat.end(), a->args.begin(), a->args.end());
    } else {
      flat.push_back(a);
    }
  }
  std::vector<const Expr*> newargs;
  std::vector<const Expr*> rels;
  for (const Expr* x : ordered(flat)) {
    if (x->is_relational()) {
      const Expr* c = canonical(x);
      if (contains(rels, c)) {
        continue;
      }
      if (contains(rels, canonical(negated(c)))) {
        return false_;
      }
      rels.push_back(c);
    }
    newargs.push_back(x);
  }
  return lattice(Kind::And, newargs);
}

const Expr* ExprArena::logical_or(c10::ArrayRef<const Expr*> args) {
  // Or._new_args_filter: unlike And, relationals are checked in argument order
  // before nested Ors are flattened.
  c10::SmallVector<const Expr*, 8> checked;
  for (const Expr* a : args) {
    checked.push_back(as_boolean(a));
  }
  std::vector<const Expr*> newargs;
  std::vector<const Expr*> rels;
  for (const Expr* x : checked) {
    if (x->is_relational()) {
      const Expr* c = canonical(x);
      if (contains(rels, c)) {
        continue;
      }
      if (contains(rels, canonical(negated(c)))) {
        return true_;
      }
      rels.push_back(c);
    }
    newargs.push_back(x);
  }
  std::vector<const Expr*> flat;
  for (const Expr* a : newargs) {
    if (a == true_) {
      return true_;
    }
    if (a == false_) {
      continue;
    }
    if (a->kind == Kind::Or) {
      flat.insert(flat.end(), a->args.begin(), a->args.end());
    } else {
      flat.push_back(a);
    }
  }
  return lattice(Kind::Or, flat);
}

const Expr* ExprArena::lattice_from_args(
    Kind kind,
    c10::ArrayRef<const Expr*> args) {
  if ((kind != Kind::And && kind != Kind::Or) || args.size() < 2) {
    throw NativeUnsupported("expected the args of an And/Or");
  }
  return intern(kind, 0, 0, args);
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

const Expr* ExprArena::canonical(const Expr* r) {
  check_relational(r);
  const Expr* a0 =
      r->args[0]->is_relational() ? canonical(r->args[0]) : r->args[0];
  const Expr* a1 =
      r->args[1]->is_relational() ? canonical(r->args[1]) : r->args[1];
  if (a0 != r->args[0] || a1 != r->args[1]) {
    r = rel(r->kind, a0, a1);
    if (!r->is_relational()) {
      return r;
    }
  }
  const Expr* lhs = r->args[0];
  const Expr* rhs = r->args[1];
  if (rhs->is_number()) {
    int il = infinity_rank(lhs);
    int ir = infinity_rank(rhs);
    if (lhs->is_number() &&
        (il != 0 || ir != 0 ? il > ir
                            : i128(lhs->p) * rhs->q > i128(rhs->p) * lhs->q)) {
      r = reversed(r);
    }
  } else if (lhs->is_number()) {
    r = reversed(r);
  } else if (ordered({lhs, rhs})[0] != lhs) {
    r = reversed(r);
  }
  lhs = r->args[0];
  rhs = r->args[1];
  if (is_boolean_atom(lhs) || is_boolean_atom(rhs)) {
    return r;
  }
  if (could_extract_minus_sign(lhs)) {
    return reversedsign(r);
  }
  if (!rhs->is_number() && could_extract_minus_sign(rhs) &&
      ordered({lhs, neg(rhs)})[0] != lhs) {
    return reversedsign(reversed(r));
  }
  return r;
}


const Expr* ExprArena::lattice_to_nnf(
    Kind kind,
    c10::ArrayRef<const Expr*> args) {
  // BooleanFunction._to_nnf(*args, simplify=False).
  std::vector<const Expr*> argset;
  for (const Expr* a : args) {
    const Expr* n = is_literal(a) ? a : to_nnf(a);
    if (!contains(argset, n)) {
      argset.push_back(n);
    }
  }
  if (kind == Kind::And) {
    return logical_and(argset);
  }
  // Or(*argset) iterates a Python set. Or drops all but the first of the
  // relationals sharing a canonical form, and its negation check only looks
  // back, which is not symmetric.
  std::vector<std::pair<const Expr*, const Expr*>> rels;
  for (const Expr* a : argset) {
    if (!a->is_relational()) {
      continue;
    }
    const Expr* c = canonical(a);
    const Expr* n = canonical(negated(c));
    for (const auto& [c2, n2] : rels) {
      if (c == c2 || (n == c2) != (n2 == c)) {
        throw NativeUnsupported("Or of relationals in set order");
      }
    }
    rels.emplace_back(c, n);
  }
  return logical_or(argset);
}

const Expr* ExprArena::to_nnf(const Expr* e) {
  // The to_nnf(simplify=False) methods.
  if (e->kind == Kind::And || e->kind == Kind::Or) {
    return lattice_to_nnf(e->kind, e->args);
  }
  if (e->kind != Kind::Not || is_literal(e)) {
    return e;
  }
  const Expr* inner = e->args[0];
  if (inner->kind != Kind::And && inner->kind != Kind::Or) {
    throw NativeUnsupported("Illegal operator in Not.to_nnf");
  }
  c10::SmallVector<const Expr*, 4> negs;
  for (const Expr* a : inner->args) {
    negs.push_back(logical_not(a));
  }
  return lattice_to_nnf(inner->kind == Kind::And ? Kind::Or : Kind::And, negs);
}

const Expr* ExprArena::distribute_and_over_or(const Expr* e) {
  if (e->kind == Kind::Or) {
    auto it = std::find_if(e->args.begin(), e->args.end(), [](const Expr* a) {
      return a->kind == Kind::And;
    });
    if (it == e->args.end()) {
      return e;
    }
    const Expr* conj = *it;
    c10::SmallVector<const Expr*, 4> others;
    for (const Expr* a : e->args) {
      if (a != conj) {
        others.push_back(a);
      }
    }
    const Expr* rest = logical_or(others);
    c10::SmallVector<const Expr*, 4> clauses;
    for (const Expr* c : conj->args) {
      clauses.push_back(distribute_and_over_or(logical_or({c, rest})));
    }
    return logical_and(clauses);
  }
  if (e->kind == Kind::And) {
    c10::SmallVector<const Expr*, 4> clauses;
    for (const Expr* a : e->args) {
      clauses.push_back(distribute_and_over_or(a));
    }
    return logical_and(clauses);
  }
  return e;
}

const Expr* ExprArena::reduce_to_lowest_terms(const Expr* e) {
  auto div_by_factor = [&](const Expr* x, int64_t factor) {
    if (x->kind == Kind::Integer) {
      return integer(x->p / factor);
    }
    c10::SmallVector<const Expr*, 4> args(x->args.begin(), x->args.end());
    // factor is 1 unless args[0] is an Integer.
    if (args[0]->kind == Kind::Integer && args[0]->p == factor) {
      args.erase(args.begin());
    } else if (args[0]->kind == Kind::Integer) {
      args[0] = integer(args[0]->p / factor);
    }
    return sorted_from_args(Kind::Mul, args);
  };
  if (e->kind == Kind::Add) {
    int64_t factor = 0;
    for (const Expr* a : e->args) {
      factor = std::gcd(factor, integer_coefficient(a));
    }
    if (factor == 1) {
      return e;
    }
    c10::SmallVector<const Expr*, 8> atoms;
    for (const Expr* a : e->args) {
      atoms.push_back(div_by_factor(a, factor));
    }
    return sorted_from_args(Kind::Add, atoms);
  }
  if (e->kind == Kind::Integer) {
    return one_;
  }
  if (e->kind == Kind::Mul) {
    return div_by_factor(e, integer_coefficient(e));
  }
  return e;
}

const Expr* ExprArena::canonicalize_bool_expr_impl(const Expr* e) {
  if (e->kind == Kind::And || e->kind == Kind::Or) {
    c10::SmallVector<const Expr*, 4> args;
    for (const Expr* a : e->args) {
      args.push_back(canonicalize_bool_expr(a));
    }
    return e->kind == Kind::And ? logical_and(args) : logical_or(args);
  }
  if (!e->is_relational()) {
    throw NativeUnsupported("Expected Lt/Le/Eq/Ne");
  }
  bool swap = e->kind == Kind::Gt || e->kind == Kind::Ge;
  Kind t = swap ? reversed_kind(e->kind) : e->kind;
  const Expr* lhs = e->args[swap ? 1 : 0];
  const Expr* rhs = e->args[swap ? 0 : 1];
  if (lhs->is_boolean() || rhs->is_boolean()) {
    return rel(
        t, canonicalize_bool_expr(lhs), canonicalize_bool_expr(rhs), false);
  }
  rhs = reduce_to_lowest_terms(sub(rhs, lhs));
  lhs = zero_;
  if (rhs->kind == Kind::Add) {
    c10::SmallVector<const Expr*, 8> pos;
    c10::SmallVector<const Expr*, 8> negs;
    for (const Expr* term : rhs->args) {
      if (is_neg(*this, term)) {
        negs.push_back(neg(term));
      } else {
        pos.push_back(term);
      }
    }
    rhs = from_args(Kind::Add, pos);
    lhs = sorted_from_args(Kind::Add, negs);
  } else if (is_neg(*this, rhs)) {
    lhs = neg(rhs);
    rhs = zero_;
  }
  return rel(t, lhs, rhs, false);
}

const Expr* ExprArena::canonicalize_bool_expr(const Expr* e) {
  if (e->kind == Kind::And || e->kind == Kind::Or || e->kind == Kind::Not) {
    // to_cnf(e): eliminate_implications is to_nnf(e, simplify=False).
    if (!is_cnf(e)) {
      e = distribute_and_over_or(is_nnf(e) ? e : to_nnf(e));
    }
  } else if (!e->is_relational()) {
    return e;
  }
  return canonicalize_bool_expr_impl(e);
}

} // namespace torch::symbolic
