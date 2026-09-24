#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

namespace torch::symbolic {

namespace {

using F = Fact;
using i128 = __int128;

bool holds(ExprArena& A, const Expr* e, Fact f) {
  return A.ask(e, f) == Tri::True;
}

bool has_symbol(const Expr* e, const Expr* x) {
  return e == x ||
      std::any_of(e->args.begin(), e->args.end(), [x](const Expr* a) {
           return has_symbol(a, x);
         });
}

bool has_pow(const Expr* e) {
  return e->kind == Kind::Pow ||
      std::any_of(e->args.begin(), e->args.end(), has_pow);
}

c10::ArrayRef<const Expr*> make_args(const Expr* const& e) {
  if (e->kind == Kind::Mul) {
    return e->args;
  }
  return c10::ArrayRef<const Expr*>(&e, 1);
}

int64_t to_int64(i128 v) {
  if (v < std::numeric_limits<int64_t>::min() ||
      v > std::numeric_limits<int64_t>::max()) {
    throw NativeUnsupported("integer overflow");
  }
  return static_cast<int64_t>(v);
}

} // namespace

std::pair<const Expr*, const Expr*> ExprArena::as_coeff_Add(const Expr* e) {
  if (e->kind == Kind::Add && e->args[0]->is_number()) {
    return {e->args[0], add(c10::ArrayRef<const Expr*>(e->args).slice(1))};
  }
  return {zero_, e};
}

c10::SmallVector<const Expr*, 4> ExprArena::free_symbols(const Expr* e) const {
  c10::SmallVector<const Expr*, 4> out;
  c10::SmallVector<const Expr*, 16> stack{e};
  while (!stack.empty()) {
    const Expr* t = stack.pop_back_val();
    if (t->kind == Kind::Symbol) {
      if (std::find(out.begin(), out.end(), t) == out.end()) {
        out.push_back(t);
      }
    }
    stack.append(t->args.begin(), t->args.end());
  }
  return out;
}

bool ExprArena::is_polynomial(const Expr* e) const {
  // Expr.is_polynomial() in all free symbols. Pow exponents are Integers.
  if (e->kind == Kind::Pow) {
    return is_polynomial(e->args[0]) && e->args[1]->p >= 0;
  }
  return std::all_of(e->args.begin(), e->args.end(), [this](const Expr* a) {
    return is_polynomial(a);
  });
}

const Expr* ExprArena::diff(const Expr* e, const Expr* x) {
  // Derivative(e, x, evaluate=True): 0 unless x is free in e.
  if (!has_symbol(e, x)) {
    return zero_;
  }
  switch (e->kind) {
    case Kind::Symbol:
      return one_;
    case Kind::Add: {
      c10::SmallVector<const Expr*, 8> terms;
      for (const Expr* a : e->args) {
        terms.push_back(diff(a, x));
      }
      return add(terms);
    }
    case Kind::Mul: {
      // Mul._eval_derivative_n_times with n = 1 (Leibniz rule).
      c10::SmallVector<const Expr*, 8> terms;
      for (size_t i = 0; i < e->args.size(); ++i) {
        c10::SmallVector<const Expr*, 8> factors(
            e->args.begin(), e->args.end());
        factors[i] = diff(e->args[i], x);
        terms.push_back(mul(factors));
      }
      return add(terms);
    }
    case Kind::Pow: {
      // self * (dexp*log(base) + dbase*exp/base) with dexp = 0.
      const Expr* b = e->args[0];
      const Expr* t = mul({diff(b, x), e->args[1]});
      return mul({e, mul({t, pow(b, neg_one_)})});
    }
    default:
      return zero_;
  }
}

const Expr* ExprArena::xreplace(
    const Expr* e,
    c10::ArrayRef<std::pair<const Expr*, const Expr*>> reps) {
  for (const auto& [old, rep] : reps) {
    if (e == old) {
      return rep;
    }
  }
  if (e->args.empty()) {
    return e;
  }
  c10::SmallVector<const Expr*, 8> args;
  bool changed = false;
  for (const Expr* a : e->args) {
    args.push_back(xreplace(a, reps));
    changed |= args.back() != a;
  }
  if (!changed) {
    return e;
  }
  if (e->is_boolean()) {
    // sympy rebuilds whenever a rule matched, even with identical args, and
    // rebuilding an unevaluated relational or an Or is not a no-op.
    throw NativeUnsupported("xreplace of a Boolean");
  }
  return e->kind == Kind::Add ? add(args)
      : e->kind == Kind::Mul  ? mul(args)
                              : pow(args[0], args[1]);
}

const Expr* ExprArena::keep_coeff(const Expr* coeff, const Expr* factors) {
  // _keep_coeff(coeff, factors) for a Rational coeff. Except for a Number times
  // an Add, which it leaves unevaluated, it equals the evaluated product.
  if (factors == one_) {
    return coeff;
  }
  if (coeff == one_) {
    return factors;
  }
  if (coeff == neg_one_) {
    return neg(factors);
  }
  if (factors->kind == Kind::Add) {
    throw NativeUnsupported("unevaluated Mul(Number, Add) in _keep_coeff");
  }
  return mul({coeff, factors});
}

std::pair<const Expr*, const Expr*> ExprArena::as_numer_denom(const Expr* e) {
  switch (e->kind) {
    case Kind::Rational:
      return {integer(e->p), integer(e->q)};
    case Kind::Mul: {
      c10::SmallVector<const Expr*, 8> nums;
      c10::SmallVector<const Expr*, 8> denoms;
      for (const Expr* f : e->args) {
        auto [n, d] = as_numer_denom(f);
        nums.push_back(n);
        denoms.push_back(d);
      }
      return {mul(nums), mul(denoms)};
    }
    case Kind::Pow: {
      // Pow.as_numer_denom with an Integer exponent.
      auto [n, d] = as_numer_denom(e->args[0]);
      if (holds(*this, d, F::nonpositive)) {
        n = neg(n);
        d = neg(d);
      }
      const Expr* x = e->args[1];
      if (x->p < 0) {
        std::swap(n, d);
        x = neg(x);
      }
      return {pow(n, x), pow(d, x)};
    }
    case Kind::Add: {
      // Add.primitive(): the Rational gcd of the leading coefficients.
      c10::SmallVector<std::pair<Num, const Expr*>, 8> terms;
      int64_t ngcd = 0;
      int64_t dlcm = 1;
      for (const Expr* a : e->args) {
        Num c{1, 1};
        const Expr* m = a;
        if (a->is_rational()) {
          c = {a->p, a->q};
          m = one_;
        } else if (a->kind == Kind::Mul && a->args[0]->is_rational()) {
          c = {a->args[0]->p, a->args[0]->q};
          m = mul(c10::ArrayRef<const Expr*>(a->args).slice(1));
        }
        if (c.p == std::numeric_limits<int64_t>::min()) {
          throw NativeUnsupported("integer overflow in Add.primitive");
        }
        terms.emplace_back(c, m);
        ngcd = std::gcd(ngcd, c.p);
        dlcm = to_int64(i128(dlcm) / std::gcd(dlcm, c.q) * c.q);
      }
      const Expr* ncon = one_;
      const Expr* dcon = one_;
      const Expr* expr = e;
      if (ngcd != 1 || dlcm != 1) {
        c10::SmallVector<const Expr*, 8> scaled;
        for (auto [c, m] : terms) {
          i128 k = i128(c.p / ngcd) * (dlcm / c.q);
          scaled.push_back(keep_coeff(integer(to_int64(k)), m));
        }
        expr = add(scaled);
        ncon = integer(ngcd);
        dcon = integer(dlcm);
      }
      // Group the numerators by denominator, in first-seen order.
      c10::SmallVector<
          std::pair<const Expr*, c10::SmallVector<const Expr*, 4>>,
          4>
          nd;
      for (const Expr* f : expr->args) {
        auto [ni, di] = as_numer_denom(f);
        auto it = std::find_if(nd.begin(), nd.end(), [di = di](const auto& g) {
          return g.first == di;
        });
        if (it == nd.end()) {
          nd.push_back({di, {}});
          it = nd.end() - 1;
        }
        it->second.push_back(ni);
      }
      if (nd.size() == 1) {
        c10::SmallVector<const Expr*, 8> ns;
        for (const Expr* ni : nd[0].second) {
          ns.push_back(keep_coeff(ncon, ni));
        }
        return {add(ns), keep_coeff(dcon, nd[0].first)};
      }
      c10::SmallVector<const Expr*, 4> denoms;
      c10::SmallVector<const Expr*, 4> nums;
      for (const auto& [d, ns] : nd) {
        denoms.push_back(d);
        nums.push_back(ns.size() > 1 ? add(ns) : ns[0]);
      }
      c10::SmallVector<const Expr*, 4> ns;
      for (size_t i = 0; i < nums.size(); ++i) {
        c10::SmallVector<const Expr*, 4> factors(denoms.begin(), denoms.end());
        factors[i] = nums[i];
        ns.push_back(mul(factors));
      }
      return {keep_coeff(ncon, add(ns)), keep_coeff(dcon, mul(denoms))};
    }
    default:
      return {e, one_};
  }
}

c10::SmallVector<const Expr*, 2> ExprArena::real_roots(
    const Expr* p,
    const Expr* x) {
  // real_roots(p) for a polynomial in x of degree <= 1 (with multiplicity).
  // Coefficients by expansion, lowest degree first.
  constexpr size_t kMaxDegree = 64;
  auto coeffs = [&](auto&& self, const Expr* e) -> std::vector<const Expr*> {
    if (e == x) {
      return {zero_, one_};
    }
    if (e->is_number()) {
      return {e};
    }
    auto product = [&](const std::vector<const Expr*>& a,
                       const std::vector<const Expr*>& b) {
      if (a.size() + b.size() > kMaxDegree + 2) {
        throw NativeUnsupported("polynomial degree too large");
      }
      std::vector<const Expr*> r(a.size() + b.size() - 1, zero_);
      for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
          r[i + j] = add({r[i + j], mul({a[i], b[j]})});
        }
      }
      return r;
    };
    switch (e->kind) {
      case Kind::Add: {
        std::vector<const Expr*> r;
        for (const Expr* a : e->args) {
          auto c = self(self, a);
          r.resize(std::max(r.size(), c.size()), zero_);
          for (size_t i = 0; i < c.size(); ++i) {
            r[i] = add({r[i], c[i]});
          }
        }
        return r;
      }
      case Kind::Mul: {
        std::vector<const Expr*> r{one_};
        for (const Expr* a : e->args) {
          r = product(r, self(self, a));
        }
        return r;
      }
      case Kind::Pow: {
        if (e->args[1]->p < 0) {
          throw NativeUnsupported("real_roots of a rational function");
        }
        auto base = self(self, e->args[0]);
        std::vector<const Expr*> r{one_};
        for (int64_t i = 0; i < e->args[1]->p; ++i) {
          r = product(r, base);
        }
        return r;
      }
      default:
        throw NativeUnsupported("real_roots of a multivariate polynomial");
    }
  };
  std::vector<const Expr*> c = coeffs(coeffs, p);
  while (!c.empty() && c.back() == zero_) {
    c.pop_back();
  }
  if (c.empty() || c.size() > 2) {
    throw NativeUnsupported("real_roots of a zero or nonlinear polynomial");
  }
  if (c.size() == 1) {
    return {};
  }
  return {mul({neg(c[0]), pow(c[1], neg_one_)})};
}

const Expr* ExprArena::monotonic_sign(const Expr* self) {
  if (!holds(*this, self, F::extended_real)) {
    return nullptr;
  }
  if (const Expr* m = neg(self); m->kind == Kind::Symbol) {
    const Expr* rv = monotonic_sign(m);
    return rv ? neg(rv) : nullptr;
  }

  if (self->kind != Kind::Add && as_numer_denom(self).second->is_number()) {
    const Expr* s = self;
    if (holds(*this, s, F::prime)) {
      return integer(holds(*this, s, F::odd) ? 3 : 2);
    } else if (holds(*this, s, F::composite)) {
      return integer(holds(*this, s, F::odd) ? 9 : 4);
    } else if (holds(*this, s, F::positive)) {
      if (holds(*this, s, F::even)) {
        return integer(ask(s, F::prime) == Tri::False ? 4 : 2);
      } else if (holds(*this, s, F::integer)) {
        return one_;
      } else {
        return eps_;
      }
    } else if (holds(*this, s, F::extended_negative)) {
      if (holds(*this, s, F::even)) {
        return integer(-2);
      } else if (holds(*this, s, F::integer)) {
        return neg_one_;
      } else {
        return neg(eps_);
      }
    }
    if (holds(*this, s, F::zero) || holds(*this, s, F::extended_nonpositive) ||
        holds(*this, s, F::extended_nonnegative)) {
      return zero_;
    }
    return nullptr;
  }

  // univariate polynomial
  auto free = free_symbols(self);
  if (free.size() == 1) {
    if (is_polynomial(self)) {
      const Expr* x = free[0];
      const Expr* x0 = monotonic_sign(x);
      if (x0 == eps_ || x0 == neg(eps_)) {
        x0 = zero_;
      }
      if (x0 != nullptr) {
        const Expr* d = diff(self, x);
        // real_roots is pure, so it is computed only when an `all` needs it.
        std::optional<c10::SmallVector<const Expr*, 2>> roots;
        auto all_roots = [&](Fact f) {
          if (!roots) {
            roots = d->is_number() ? c10::SmallVector<const Expr*, 2>{}
                                   : real_roots(d, x);
          }
          return std::all_of(roots->begin(), roots->end(), [&](const Expr* r) {
            return holds(*this, sub(r, x0), f);
          });
        };
        std::pair<const Expr*, const Expr*> rep{x, x0};
        const Expr* y = xreplace(self, rep);
        if (holds(*this, x, F::nonnegative) && all_roots(F::nonpositive)) {
          if (holds(*this, y, F::nonnegative) && holds(*this, d, F::positive)) {
            if (y != zero_) {
              return holds(*this, y, F::positive) ? y
                                                  : dummy("pos", F::positive);
            }
            return dummy("nneg", F::nonnegative);
          }
          if (holds(*this, y, F::nonpositive) && holds(*this, d, F::negative)) {
            if (y != zero_) {
              return holds(*this, y, F::negative) ? y
                                                  : dummy("neg", F::negative);
            }
            return dummy("npos", F::nonpositive);
          }
        } else if (
            holds(*this, x, F::nonpositive) && all_roots(F::nonnegative)) {
          if (holds(*this, y, F::nonnegative) && holds(*this, d, F::negative)) {
            return y != zero_ ? dummy("pos", F::positive)
                              : dummy("nneg", F::nonnegative);
          }
          if (holds(*this, y, F::nonpositive) && holds(*this, d, F::positive)) {
            return y != zero_ ? dummy("neg", F::negative)
                              : dummy("npos", F::nonpositive);
          }
        }
      }
    } else {
      auto [n, d] = as_numer_denom(self);
      const Expr* den = nullptr;
      if (n->is_number()) {
        den = monotonic_sign(d);
      } else if (!d->is_number()) {
        if (monotonic_sign(n) != nullptr) {
          den = monotonic_sign(d);
        }
      }
      if (den != nullptr &&
          (holds(*this, den, F::positive) || holds(*this, den, F::negative))) {
        const Expr* v = mul({n, den});
        if (holds(*this, v, F::positive)) {
          return dummy("pos", F::positive);
        } else if (holds(*this, v, F::nonnegative)) {
          return dummy("nneg", F::nonnegative);
        } else if (holds(*this, v, F::negative)) {
          return dummy("neg", F::negative);
        } else if (holds(*this, v, F::nonpositive)) {
          return dummy("npos", F::nonpositive);
        }
      }
    }
    return nullptr;
  }

  // multivariate
  auto [c, a] = as_coeff_Add(self);
  const Expr* v = nullptr;
  if (!is_polynomial(a)) {
    // An Add `a` never sets v below, so sympy returns None either way.
    if (a->kind == Kind::Add) {
      return nullptr;
    }
    auto [n, d] = as_numer_denom(a);
    if (!(n->is_number() || d->is_number())) {
      return nullptr;
    }
    if (holds(*this, a, F::rational) &&
        (holds(*this, a, F::positive) || holds(*this, a, F::negative))) {
      v = one_;
      for (const Expr* ai : make_args(a)) {
        if (ai->is_number()) {
          v = mul({v, ai});
          continue;
        }
        c10::SmallVector<std::pair<const Expr*, const Expr*>, 4> reps;
        for (const Expr* x : free_symbols(ai)) {
          const Expr* r = monotonic_sign(x);
          if (r == nullptr) {
            return nullptr;
          }
          reps.emplace_back(x, r);
        }
        v = mul({v, xreplace(ai, reps)});
      }
    }
  } else if (c != zero_) {
    // signed linear expression
    if (!has_pow(a) &&
        (holds(*this, a, F::nonpositive) || holds(*this, a, F::nonnegative))) {
      c10::SmallVector<std::pair<const Expr*, const Expr*>, 4> reps;
      for (const Expr* x : free_symbols(a)) {
        const Expr* r = monotonic_sign(x);
        if (r == nullptr) {
          return nullptr;
        }
        if (r == zero_) {
          r = holds(*this, x, F::nonnegative) ? eps_ : neg(eps_);
        }
        reps.emplace_back(x, r);
      }
      v = xreplace(a, reps);
    }
  }
  if (v != nullptr) {
    const Expr* rv = add({v, c});
    if ((holds(*this, v, F::nonnegative) && holds(*this, rv, F::positive)) ||
        (holds(*this, v, F::nonpositive) && holds(*this, rv, F::negative))) {
      std::pair<const Expr*, const Expr*> rep{eps_, zero_};
      return xreplace(rv, rep);
    }
  }
  return nullptr;
}

} // namespace torch::symbolic
