#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <limits>

// Port of safe_expand from torch/fx/experimental/symbolic_shapes.py.

namespace torch::symbolic {

namespace {

int64_t binomial(int64_t n, int64_t k) {
  k = std::min(k, n - k);
  __int128 v = 1;
  for (int64_t j = 0; j < k; ++j) {
    v = v * (n - j) / (j + 1);
    if (v > std::numeric_limits<int64_t>::max()) {
      throw NativeUnsupported("integer overflow");
    }
  }
  return static_cast<int64_t>(v);
}

} // namespace

const Expr* ExprArena::rebuild(
    const Expr* e,
    c10::ArrayRef<const Expr*> args) {
  switch (e->kind) {
    case Kind::Add:
      return add(args);
    case Kind::Mul:
      return mul(args);
    case Kind::Pow:
      return pow(args[0], args[1]);
    case Kind::Not:
      return logical_not(args[0]);
    case Kind::And:
      return logical_and(args);
    case Kind::Or:
      return logical_or(args);
    default:
      break;
  }
  if (e->is_relational()) {
    return rel(e->kind, args[0], args[1]);
  }
  if (e->is_function()) {
    return function(e->kind, args);
  }
  throw NativeUnsupported("rebuild of an atom");
}

const Expr* ExprArena::expand_multinomial(const Expr* e) {
  // expand_multinomial(e, deep=False): Expr.expand applies
  // Pow._eval_expand_multinomial until the expression stops changing; no other
  // implemented kind has that hint.
  while (e->kind == Kind::Pow && e->args[0]->kind == Kind::Add &&
         e->args[1]->kind == Kind::Integer) {
    const Expr* base = e->args[0];
    int64_t n = e->args[1]->p;
    const Expr* r = e;
    if (n < -1) {
      r = pow(expand_multinomial(pow(base, integer(-n))), neg_one_);
    } else if (n > 1) {
      // basic_from_dict(multinomial_coefficients(len(p), n), *p) with p the
      // terms of base: the order of the terms does not change the Add.
      auto p = base->args;
      std::vector<const Expr*> terms;
      c10::SmallVector<const Expr*, 8> term;
      std::vector<int64_t> k(p.size());
      auto emit = [&](auto& self, size_t i, int64_t left, int64_t coeff) {
        if (i + 1 == p.size()) {
          k[i] = left;
          term.assign({integer(coeff)});
          for (size_t j = 0; j < p.size(); ++j) {
            if (k[j] != 0) {
              term.push_back(pow(p[j], integer(k[j])));
            }
          }
          terms.push_back(mul(term));
          return;
        }
        for (int64_t ki = left; ki >= 0; --ki) {
          int64_t c = 0;
          if (__builtin_mul_overflow(coeff, binomial(left, ki), &c)) {
            throw NativeUnsupported("integer overflow");
          }
          k[i] = ki;
          self(self, i + 1, left - ki, c);
        }
      };
      emit(emit, 0, n, 1);
      r = add(terms);
    }
    if (r == e) {
      break;
    }
    e = r;
  }
  return e;
}

std::pair<const Expr*, bool> ExprArena::expandsums(
    c10::ArrayRef<const Expr*> args) {
  // _expandsums.
  std::vector<const Expr*> adds;
  std::vector<const Expr*> other;
  for (const Expr* a : args) {
    (a->kind == Kind::Add ? adds : other).push_back(a);
  }
  std::vector<const Expr*> result = {mul(other)};
  for (const Expr* s : adds) {
    std::vector<const Expr*> next;
    for (const Expr* a : result) {
      for (const Expr* b : s->args) {
        next.push_back(mul({a, b}));
      }
    }
    result = std::move(next);
  }
  return {add(result), adds.size() > 1 || (!adds.empty() && !other.empty())};
}

const Expr* ExprArena::fast_expand(const Expr* e) {
  // _fast_expand. Python rebuilds when a child is a new object; hash-consing
  // makes that pointer inequality, which differs only where Python rebuilds an
  // equal child without sympy's cache returning the old object.
  if (!e->args.empty()) {
    c10::SmallVector<const Expr*, 8> args;
    bool changed = false;
    for (const Expr* a : e->args) {
      args.push_back(fast_expand(a));
      changed |= args.back() != a;
    }
    if (changed) {
      return fast_expand(rebuild(e, args));
    }
  }
  if (e->kind == Kind::Pow && e->args[0]->kind == Kind::Add &&
      e->args[1]->kind == Kind::Integer) {
    if (e->args[1]->p > 1) {
      return expand_multinomial(e);
    }
    if (e->args[1]->p < 0) {
      return pow(expand_multinomial(pow(e, neg_one_)), neg_one_);
    }
  } else if (e->kind == Kind::Mul) {
    std::vector<const Expr*> num;
    std::vector<const Expr*> den;
    for (const Expr* a : e->args) {
      if (a->kind == Kind::Pow && a->args[1] == neg_one_) {
        den.push_back(pow(a, neg_one_));
      } else {
        num.push_back(a);
      }
    }
    auto [n, num_changed] = expandsums(num);
    auto [d, den_changed] = expandsums(den);
    if (num_changed || den_changed) {
      return mul({n, pow(d, neg_one_)});
    }
  }
  return e;
}

const Expr* ExprArena::safe_expand(const Expr* e) {
  // And, Or and Not have no expand method.
  if (e->kind == Kind::And || e->kind == Kind::Or || e->kind == Kind::Not) {
    return e;
  }
  return fast_expand(e);
}

} // namespace torch::symbolic
