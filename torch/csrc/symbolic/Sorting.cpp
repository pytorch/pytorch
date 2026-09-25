#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <bit>
#include <cmath>

namespace torch::symbolic {

namespace {

using i128 = __int128;
using T = SortKey::Type;

template <typename V>
int cmp3(V a, V b) {
  return (a > b) - (a < b);
}

SortKeyPtr key_int(int64_t v) {
  auto k = std::make_shared<SortKey>(SortKey{T::Int});
  k->i = v;
  return k;
}

SortKeyPtr key_str(std::string v) {
  auto k = std::make_shared<SortKey>(SortKey{T::Str});
  k->s = std::move(v);
  return k;
}

SortKeyPtr key_num(const Expr* e) {
  auto k = std::make_shared<SortKey>(SortKey{T::Num});
  k->num = e;
  return k;
}

SortKeyPtr key_tuple(std::vector<SortKeyPtr> items) {
  auto k = std::make_shared<SortKey>(SortKey{T::Tuple});
  k->items = std::move(items);
  return k;
}

// -oo < -int_oo < finite numbers < int_oo < oo.
int infinity_rank(const Expr* e) {
  switch (e->kind) {
    case Kind::Infinity:
      return 2;
    case Kind::IntInfinity:
      return 1;
    case Kind::NegativeIntInfinity:
      return -1;
    case Kind::NegativeInfinity:
      return -2;
    default:
      return 0;
  }
}

bool is_negative_number(const Expr* e) {
  return e->kind == Kind::NegativeIntInfinity ||
      e->kind == Kind::NegativeInfinity ||
      (e->kind == Kind::Float ? e->float_value() < 0
                              : e->is_rational() && e->p < 0);
}

int bit_width(unsigned __int128 x) {
  auto hi = static_cast<uint64_t>(x >> 64);
  return hi != 0 ? 64 + std::bit_width(hi)
                 : std::bit_width(static_cast<uint64_t>(x));
}

// Float._Frel with a Rational: v*q against p, exactly.
int compare_float_rational(double v, int64_t p, int64_t q) {
  int c = cmp3(v > 0 ? 1 : v < 0 ? -1 : 0, cmp3(p, int64_t(0)));
  if (c != 0 || p == 0) {
    return c;
  }
  // |v|*q = m*q*2**e with m < 2**53, against |p| < 2**64.
  int e = 0;
  auto m = static_cast<uint64_t>(std::ldexp(std::frexp(std::abs(v), &e), 53));
  e -= 53;
  unsigned __int128 lhs = static_cast<unsigned __int128>(m) * q;
  unsigned __int128 rhs = p < 0 ? -static_cast<uint64_t>(p) : p;
  if (e >= 0) {
    c = bit_width(lhs) + e > 64 ? 1 : cmp3(lhs << e, rhs);
  } else {
    c = bit_width(rhs) - e > 120 ? -1 : cmp3(lhs, rhs << -e);
  }
  return p < 0 ? -c : c;
}

const char* class_name(Kind k) {
  switch (k) {
    case Kind::Mul:
      return "Mul";
    case Kind::Add:
      return "Add";
    case Kind::Pow:
      return "Pow";
    case Kind::BooleanTrue:
      return "BooleanTrue";
    case Kind::BooleanFalse:
      return "BooleanFalse";
    case Kind::Eq:
      return "Equality";
    case Kind::Ne:
      return "Unequality";
    case Kind::Lt:
      return "StrictLessThan";
    case Kind::Le:
      return "LessThan";
    case Kind::Gt:
      return "StrictGreaterThan";
    case Kind::Ge:
      return "GreaterThan";
    case Kind::Not:
      return "Not";
    case Kind::And:
      return "And";
    case Kind::Or:
      return "Or";
    default:
      return function_name(k);
  }
}

// sympy's _node_count doubled: a Float counts as half a node.
size_t node_count(const Expr* e) {
  if (e->kind == Kind::Float) {
    return 1;
  }
  size_t n = 2;
  for (const Expr* a : e->args) {
    n += node_count(a);
  }
  return n;
}

} // namespace

// Number.__lt__ / __gt__, including int_oo's overloads and oo.
int compare_numbers(const Expr* a, const Expr* b) {
  int ra = infinity_rank(a);
  int rb = infinity_rank(b);
  if (ra != 0 || rb != 0) {
    return cmp3(ra, rb);
  }
  if (a->kind == Kind::Float) {
    return b->kind == Kind::Float
        ? cmp3(a->float_value(), b->float_value())
        : compare_float_rational(a->float_value(), b->p, b->q);
  }
  if (b->kind == Kind::Float) {
    return -compare_float_rational(b->float_value(), a->p, a->q);
  }
  return cmp3(i128(a->p) * b->q, i128(b->p) * a->q);
}

int compare_keys(const SortKey& a, const SortKey& b) {
  if (a.type != b.type) {
    throw NativeUnsupported("unorderable sort keys");
  }
  switch (a.type) {
    case T::Int:
      return cmp3(a.i, b.i);
    case T::Str:
      return cmp3(a.s.compare(b.s), 0);
    case T::Num: {
      int c = compare_numbers(a.num, b.num);
      // Python stops a tuple comparison at Float(0.5) != Rational(1, 2), which
      // then compares neither less nor greater.
      if (c == 0 && a.num != b.num) {
        throw NativeUnsupported("sort keys tie on a Float and a Rational");
      }
      return c;
    }
    case T::Tuple:
      break;
  }
  size_t n = std::min(a.items.size(), b.items.size());
  for (size_t i = 0; i < n; ++i) {
    if (a.items[i] == b.items[i]) {
      continue;
    }
    int c = compare_keys(*a.items[i], *b.items[i]);
    if (c != 0) {
      return c;
    }
  }
  return cmp3(a.items.size(), b.items.size());
}

std::pair<const Expr*, const Expr*> ExprArena::as_coeff_Mul(const Expr* e) {
  if (e->is_number()) {
    return {e, one_};
  }
  if (e->kind == Kind::Mul && e->args[0]->is_number()) {
    if (e->args.size() == 2) {
      return {e->args[0], e->args[1]};
    }
    return {
        e->args[0],
        intern(Kind::Mul, 0, 0, c10::ArrayRef<const Expr*>(e->args).slice(1))};
  }
  return {one_, e};
}

const SortKeyPtr& ExprArena::sort_key(const Expr* e) {
  if (auto it = sort_keys_.find(e); it != sort_keys_.end()) {
    return it->second;
  }
  auto class_key = [](const Expr* x) {
    if (x->is_number()) {
      return key_tuple({key_int(1), key_int(0), key_str("Number")});
    }
    if (x->kind == Kind::Symbol) {
      return key_tuple({key_int(2), key_int(0), key_str("Symbol")});
    }
    // Function.class_key: nargs is a FiniteSet for every function kind but
    // IsNonOverlappingAndDenseIndicator and Identity (no eval). Max and Min
    // are not Functions and use Basic.class_key.
    bool minmax = x->kind == Kind::Max || x->kind == Kind::Min;
    bool function = x->is_function() && !minmax;
    int major = x->is_boolean() || minmax ? 5 : function ? 4 : 3;
    bool variadic = x->kind == Kind::IsNonOverlappingAndDenseIndicator ||
        x->kind == Kind::Identity;
    int minor = function       ? (variadic ? 0 : 10000)
        : x->kind == Kind::Add ? 1
        : x->kind == Kind::Pow ? 2
                               : 0;
    return key_tuple(
        {key_int(major), key_int(minor), key_str(class_name(x->kind))});
  };
  auto args_key = [](std::vector<SortKeyPtr> items) {
    int64_t n = static_cast<int64_t>(items.size());
    return key_tuple({key_int(n), key_tuple(std::move(items))});
  };

  auto check_not_dummy = [this](const Expr* x) {
    if (x->kind == Kind::Symbol && symbol_info(x).dummy_index != 0) {
      // Dummy keys include sympy's global Dummy counter.
      throw NativeUnsupported("sort_key of a Dummy");
    }
  };
  check_not_dummy(e);
  SortKeyPtr key;
  if (e->is_number()) {
    key = key_tuple(
        {class_key(e),
         key_tuple({key_int(0), key_tuple({})}),
         key_tuple({}),
         key_num(e)});
  } else if (e->kind == Kind::Symbol || e->is_boolean()) {
    // Symbol.sort_key and Basic.sort_key.
    std::vector<SortKeyPtr> items;
    if (e->kind == Kind::Symbol) {
      items.push_back(key_str(symbol_info(e).name));
    }
    for (const Expr* a : e->args) {
      items.push_back(sort_key(a));
    }
    key = key_tuple(
        {class_key(e),
         args_key(std::move(items)),
         sort_key(one_),
         key_num(one_)});
  } else {
    // Expr.sort_key; Pow bases are never Numbers.
    auto [coeff, expr] = as_coeff_Mul(e);
    const Expr* exp = one_;
    if (expr->kind == Kind::Pow) {
      exp = expr->args[1];
      expr = expr->args[0];
    }
    check_not_dummy(expr);
    std::vector<SortKeyPtr> items;
    if (expr->kind == Kind::Symbol) {
      items.push_back(key_str(symbol_info(expr).name));
    } else {
      c10::SmallVector<const Expr*, 4> args(expr->args.begin(), expr->args.end());
      if (expr->kind == Kind::Add) {
        args = as_ordered_terms(expr);
      } else if (expr->kind == Kind::Mul) {
        args = as_ordered_factors(expr);
      }
      for (const Expr* a : args) {
        items.push_back(sort_key(a));
      }
    }
    key = key_tuple(
        {class_key(expr),
         args_key(std::move(items)),
         sort_key(exp),
         key_num(coeff)});
  }
  return sort_keys_.emplace(e, std::move(key)).first->second;
}

std::vector<const Expr*> ExprArena::ordered(c10::ArrayRef<const Expr*> seq) {
  // Group by node count, then break ties with default_sort_key; keys are only
  // computed for groups of more than one element, as sympy does.
  std::vector<std::pair<size_t, const Expr*>> by_nodes;
  for (const Expr* e : seq) {
    by_nodes.emplace_back(node_count(e), e);
  }
  std::stable_sort(
      by_nodes.begin(), by_nodes.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
  std::vector<const Expr*> out;
  for (size_t i = 0; i < by_nodes.size();) {
    size_t j = i;
    while (j < by_nodes.size() && by_nodes[j].first == by_nodes[i].first) {
      ++j;
    }
    std::vector<std::pair<SortKeyPtr, const Expr*>> group;
    for (size_t k = i; k < j; ++k) {
      const Expr* e = by_nodes[k].second;
      group.emplace_back(j - i > 1 ? sort_key(e) : nullptr, e);
    }
    std::stable_sort(
        group.begin(), group.end(), [](const auto& a, const auto& b) {
          return compare_keys(*a.first, *b.first) < 0;
        });
    for (auto& g : group) {
      out.push_back(g.second);
    }
    i = j;
  }
  return out;
}

std::vector<const Expr*> ExprArena::ordered_frozenset(
    c10::ArrayRef<const Expr*> seq) {
  std::vector<const Expr*> unique;
  for (const Expr* e : seq) {
    if (std::find(unique.begin(), unique.end(), e) == unique.end()) {
      unique.push_back(e);
    }
  }
  std::vector<const Expr*> sorted = ordered(unique);
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (node_count(sorted[i - 1]) == node_count(sorted[i]) &&
        compare_keys(*sort_key(sorted[i - 1]), *sort_key(sorted[i])) == 0) {
      throw NativeUnsupported("frozenset elements with equal sort keys");
    }
  }
  return sorted;
}

c10::SmallVector<const Expr*, 4> ExprArena::as_ordered_factors(const Expr* e) {
  if (e->kind != Kind::Mul) {
    return {e};
  }
  // args_cnc(split_1=True).
  c10::SmallVector<const Expr*, 4> cpart(e->args.begin(), e->args.end());
  const Expr* c = cpart[0];
  if (c->is_number() && is_negative_number(c) && c != neg_one_) {
    cpart[0] = neg(c);
    cpart.insert(cpart.begin(), neg_one_);
  }
  std::vector<std::pair<SortKeyPtr, const Expr*>> keyed;
  for (const Expr* f : cpart) {
    keyed.emplace_back(sort_key(f), f);
  }
  std::stable_sort(
      keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
        return compare_keys(*a.first, *b.first) < 0;
      });
  c10::SmallVector<const Expr*, 4> out;
  for (auto& k : keyed) {
    out.push_back(k.second);
  }
  return out;
}

c10::SmallVector<const Expr*, 4> ExprArena::as_ordered_terms(const Expr* e) {
  if (e->kind != Kind::Add) {
    return {e};
  }
  // Add(positive Number, Mul(negative Number, x)) keeps the number first. The
  // only Number in an Add or Mul is its first arg.
  const Expr* m = e->args.size() == 2 ? e->args[1] : nullptr;
  if (m != nullptr && e->args[0]->is_number() && m->kind == Kind::Mul &&
      m->args.size() == 2 && m->args[0]->is_number() &&
      compare_numbers(e->args[0], zero_) > 0 &&
      is_negative_number(m->args[0])) {
    return {e->args[0], m};
  }

  // Expr.as_terms with the lex key of Expr._parse_order.
  struct Term {
    const Expr* term;
    double re;
    std::vector<std::pair<const Expr*, int64_t>> cpart;
    std::vector<int64_t> monom;
  };
  std::vector<Term> terms;
  std::vector<const Expr*> gens;
  for (const Expr* t : e->args) {
    auto [coeff, rest] = as_coeff_Mul(t);
    if (!coeff->is_rational() && coeff->kind != Kind::Float) {
      throw NativeUnsupported("complex() of int_oo");
    }
    Term term{
        t,
        coeff->kind == Kind::Float
            ? coeff->float_value()
            : static_cast<double>(
                  static_cast<long double>(coeff->p) / coeff->q),
        {},
        {}};
    if (rest != one_) {
      c10::ArrayRef<const Expr*> factors = rest->kind == Kind::Mul
          ? c10::ArrayRef<const Expr*>(rest->args)
          : c10::ArrayRef<const Expr*>(&rest, 1);
      for (const Expr* f : factors) {
        if (f->is_number()) {
          throw NativeUnsupported("Number factor after the coefficient");
        }
        // decompose_power; Pow exponents are Integers.
        const Expr* base = f->kind == Kind::Pow ? f->args[0] : f;
        int64_t exp = f->kind == Kind::Pow ? f->args[1]->p : 1;
        term.cpart.emplace_back(base, exp);
        if (std::find(gens.begin(), gens.end(), base) == gens.end()) {
          gens.push_back(base);
        }
      }
    }
    terms.push_back(std::move(term));
  }
  std::vector<std::pair<SortKeyPtr, const Expr*>> keyed;
  for (const Expr* g : gens) {
    keyed.emplace_back(sort_key(g), g);
  }
  std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
    return compare_keys(*a.first, *b.first) < 0;
  });
  for (size_t i = 1; i < keyed.size(); ++i) {
    // sympy sorts a set here, so tied gens come out in hash order.
    if (compare_keys(*keyed[i - 1].first, *keyed[i].first) == 0) {
      throw NativeUnsupported("generators with equal sort keys");
    }
  }
  for (Term& t : terms) {
    t.monom.assign(keyed.size(), 0);
    for (auto [base, exp] : t.cpart) {
      for (size_t i = 0; i < keyed.size(); ++i) {
        if (keyed[i].second == base) {
          t.monom[i] = exp;
        }
      }
    }
  }
  std::stable_sort(
      terms.begin(), terms.end(), [](const Term& a, const Term& b) {
        if (a.monom != b.monom) {
          return a.monom > b.monom;
        }
        return a.re < b.re;
      });
  c10::SmallVector<const Expr*, 4> out;
  for (const Term& t : terms) {
    out.push_back(t.term);
  }
  return out;
}

bool ExprArena::could_extract_minus_sign(const Expr* e) {
  switch (e->kind) {
    case Kind::Integer:
    case Kind::Rational:
    case Kind::Float:
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
    case Kind::Infinity:
    case Kind::NegativeInfinity:
      return is_negative_number(e);
    case Kind::Mul:
      // self == -self only for zoo factors, which the arena cannot hold.
      return e->args[0]->is_number() && is_negative_number(e->args[0]);
    case Kind::Add: {
      // _could_extract_minus_sign.
      size_t negative = 0;
      for (const Expr* a : e->args) {
        negative += could_extract_minus_sign(a);
      }
      size_t positive = e->args.size() - negative;
      if (positive != negative) {
        return positive < negative;
      }
      return compare_keys(*sort_key(e), *sort_key(neg(e))) < 0;
    }
    default:
      return false;
  }
}

} // namespace torch::symbolic
