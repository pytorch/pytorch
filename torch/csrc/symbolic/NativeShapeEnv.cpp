#include <torch/csrc/symbolic/NativeShapeEnv.h>

#include <c10/util/hash.h>

#include <algorithm>
#include <atomic>
#include <string>

namespace torch::symbolic {

namespace {

bool is_int_oo(const Expr* e) {
  return e->kind == Kind::IntInfinity || e->kind == Kind::NegativeIntInfinity;
}

// expr.atoms(*kinds).
c10::SmallVector<const Expr*, 4> atoms(
    const Expr* e,
    std::initializer_list<Kind> kinds) {
  c10::SmallVector<const Expr*, 4> out;
  c10::SmallVector<const Expr*, 16> stack{e};
  while (!stack.empty()) {
    const Expr* t = stack.pop_back_val();
    if (std::find(kinds.begin(), kinds.end(), t->kind) != kinds.end() &&
        std::find(out.begin(), out.end(), t) == out.end()) {
      out.push_back(t);
    }
    stack.append(t->args.begin(), t->args.end());
  }
  return out;
}

bool is_nonnegative_value(const Expr* value) {
  return value->kind == Kind::IntInfinity ||
      (value->kind != Kind::NegativeIntInfinity && value->p >= 0);
}

std::atomic<uint64_t> query_seq{0};
std::atomic<int64_t> pending_queries{0};

} // namespace

bool& suppress_guards_tls() {
  thread_local bool value = false;
  return value;
}

NativeShapeEnv::~NativeShapeEnv() {
  pending_queries -= static_cast<int64_t>(queries_.size());
}

size_t NativeShapeEnv::QueryHash::operator()(const NativeQuery& q) const {
  return c10::get_hash(
      q.expr,
      q.evaluate,
      q.hint.index(),
      std::visit(
          [](auto v) -> int64_t {
            if constexpr (std::is_same_v<decltype(v), std::monostate>) {
              return 0;
            } else {
              return static_cast<int64_t>(v);
            }
          },
          q.hint),
      q.fallback_value.has_value(),
      q.fallback_value.value_or(false),
      q.suppress_guards);
}

void NativeShapeEnv::log_query(const NativeQuery& q, const Expr* result) {
  auto [it, inserted] = queries_.try_emplace(q);
  if (inserted) {
    ++pending_queries;
  }
  it->second = {++query_seq, result};
}

std::vector<LoggedQuery> NativeShapeEnv::take_queries() {
  std::vector<LoggedQuery> out;
  out.reserve(queries_.size());
  for (const auto& [q, v] : queries_) {
    out.push_back({v.first, q, v.second});
  }
  pending_queries -= static_cast<int64_t>(queries_.size());
  queries_.clear();
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.seq < b.seq;
  });
  return out;
}

bool NativeShapeEnv::queries_pending() {
  return pending_queries > 0;
}

void NativeShapeEnv::add_symbol(
    const Expr* sym,
    std::optional<int64_t> hint,
    const ValueRanges& range,
    bool size_like) {
  if (sym->kind != Kind::Symbol || range.is_bool() ||
      arena_->symbol_info(sym).dummy_index != 0) {
    throw NativeUnsupported("add_symbol needs an integer symbol");
  }
  TORCH_CHECK(var_to_range_.count(sym) == 0, "symbol already mirrored");
  var_to_range_.insert_or_assign(sym, range);
  if (hint) {
    backed_var_to_val_[sym] = *hint;
  }
  if (size_like) {
    size_like_.insert(sym);
  }
}

void NativeShapeEnv::update_range(const Expr* sym, const ValueRanges& range) {
  pristine_ = false;
  if (range.is_bool()) {
    var_to_range_.erase(sym);
    throw NativeUnsupported("update_range needs an integer range");
  }
  var_to_range_.insert_or_assign(sym, range);
}

std::optional<std::tuple<std::optional<int64_t>, ValueRanges, bool>>
NativeShapeEnv::mirrored(const Expr* sym) const {
  auto it = var_to_range_.find(sym);
  if (it == var_to_range_.end()) {
    return std::nullopt;
  }
  auto hint = backed_var_to_val_.find(sym);
  return std::make_tuple(
      hint == backed_var_to_val_.end() ? std::nullopt
                                       : std::optional<int64_t>(hint->second),
      it->second,
      size_like_.count(sym) != 0);
}

bool NativeShapeEnv::all_symbols_mirrored(const Expr* e) const {
  for (const Expr* s : arena_->free_symbols(e)) {
    if (var_to_range_.count(s) == 0) {
      return false;
    }
  }
  return true;
}

std::optional<const Expr*> NativeShapeEnv::lower_bound(const Expr* e) {
  ExprArena& a = *arena_;
  auto term_bound = [&](const Expr* term) -> std::optional<const Expr*> {
    if (term->is_number()) {
      return term;
    }
    auto [coeff, base] = a.as_coeff_Mul(term);
    if (base->kind != Kind::Symbol) {
      return std::nullopt;
    }
    auto it = var_to_range_.find(base);
    if (it != var_to_range_.end()) {
      bool nonneg = is_nonnegative_value(coeff);
      const Expr* bound = nonneg ? it->second.lower : it->second.upper;
      if (is_int_oo(bound)) {
        bool positive = (bound->kind == Kind::IntInfinity) == nonneg;
        return positive ? a.int_oo() : a.neg_int_oo();
      }
      return a.mul({coeff, bound});
    }
    if (a.ask(base, Fact::nonnegative) == Tri::True &&
        is_nonnegative_value(coeff)) {
      return a.integer(0);
    }
    return std::nullopt;
  };
  if (e->kind != Kind::Add) {
    return term_bound(e);
  }
  const Expr* result = a.integer(0);
  for (const Expr* term : e->args) {
    auto b = term_bound(term);
    if (!b) {
      return std::nullopt;
    }
    if (is_int_oo(*b) || is_int_oo(result)) {
      if (is_int_oo(*b) && is_int_oo(result) && *b != result) {
        throw NativeUnsupported("int_oo + -int_oo");
      }
      result = is_int_oo(*b) ? *b : result;
    } else {
      result = a.add({result, *b});
    }
  }
  return result;
}

bool NativeShapeEnv::is_nonneg_term(const Expr* term) {
  // is_nonnegative_term in ShapeEnv.simplify.
  if (term->is_number()) {
    return is_nonnegative_value(term);
  }
  if (arena_->ask(term, Fact::nonnegative) == Tri::True) {
    return true;
  }
  if (term->kind == Kind::Symbol) {
    auto it = var_to_range_.find(term);
    return it != var_to_range_.end() && is_nonnegative_value(it->second.lower);
  }
  return false;
}

bool NativeShapeEnv::definitely_le(
    const Expr* a,
    const Expr* b,
    bool use_static_fallback,
    LeCache& le_cache) {
  auto key = std::make_tuple(a, b, use_static_fallback);
  auto it = le_cache.find(key);
  if (it != le_cache.end()) {
    return it->second;
  }
  auto answer = [&](bool r) {
    le_cache.emplace(key, r);
    return r;
  };
  ExprArena& ar = *arena_;
  if (a == b) {
    return answer(true);
  }
  const Expr* diff = ar.sub(b, a);
  auto diff_lower = lower_bound(diff);
  if (diff_lower && is_nonnegative_value(*diff_lower)) {
    return answer(true);
  }
  bool nonneg_sum = diff->kind == Kind::Add
      ? std::all_of(
            diff->args.begin(),
            diff->args.end(),
            [&](const Expr* t) { return is_nonneg_term(t); })
      : is_nonneg_term(diff);
  if (nonneg_sum) {
    return answer(true);
  }
  // Every free symbol has a mirrored range, so the comparison ranges are the
  // mirror restricted to diff's symbols, and none are looked up in the
  // TracingContext.
  if (!ar.free_symbols(diff).empty()) {
    ValueRanges r = bound_sympy(ar, ar.safe_expand(diff), var_to_range_);
    if (is_nonnegative_value(r.lower)) {
      return answer(true);
    }
  }
  if (use_static_fallback) {
    const Expr* comparison = ar.rel(Kind::Le, a, b);
    if (comparison->kind == Kind::BooleanTrue) {
      return answer(true);
    }
    if (comparison->kind == Kind::BooleanFalse) {
      return answer(false);
    }
    return answer(maybe_evaluate_static(comparison) == ar.boolean(true));
  }
  return answer(false);
}

const Expr* NativeShapeEnv::simplify(const Expr* e) {
  if (!pristine_ || !all_symbols_mirrored(e)) {
    throw NativeUnsupported("simplify needs a pristine, mirrored env");
  }
  ExprArena& a = *arena_;
  // replace() is the identity while there are no replacements.
  e = a.safe_expand(e);

  LeCache le_cache;
  c10::SmallVector<std::pair<const Expr*, const Expr*>, 4> min_max_reps;
  for (const Expr* atom : atoms(e, {Kind::Min, Kind::Max})) {
    const auto& args = atom->args;
    if (args.size() < 2) {
      continue;
    }
    bool is_min = atom->kind == Kind::Min;
    bool legacy_fallback = !is_min && args.size() == 2;
    auto small = [&](const Expr* x) {
      return legacy_fallback && (x == a.integer(0) || x == a.integer(1));
    };
    std::vector<bool> keep(args.size(), true);
    for (size_t i = 0; i < args.size(); ++i) {
      if (!keep[i]) {
        continue;
      }
      for (size_t j = i + 1; j < args.size(); ++j) {
        if (!keep[j]) {
          continue;
        }
        if (definitely_le(args[i], args[j], small(args[i]), le_cache)) {
          if (is_min) {
            keep[j] = false;
          } else {
            keep[i] = false;
            break;
          }
        } else if (definitely_le(args[j], args[i], small(args[j]), le_cache)) {
          if (is_min) {
            keep[i] = false;
            break;
          } else {
            keep[j] = false;
          }
        }
      }
    }
    c10::SmallVector<const Expr*, 4> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      if (keep[i]) {
        new_args.push_back(args[i]);
      }
    }
    if (new_args.size() != args.size()) {
      min_max_reps.emplace_back(
          atom, a.minmax(atom->kind, new_args, /*evaluate=*/false));
    }
  }
  if (!min_max_reps.empty()) {
    e = a.xreplace(e, min_max_reps);
  }

  c10::SmallVector<std::pair<const Expr*, const Expr*>, 4> trunc_reps;
  for (const Expr* atom : atoms(e, {Kind::TruncToInt})) {
    const Expr* arg = atom->args[0];
    if (arg->kind == Kind::IntTrueDiv) {
      const Expr* base = arg->args[0];
      const Expr* divisor = arg->args[1];
      bool exact = a.function(Kind::Mod, {base, divisor}) == a.integer(0);
      trunc_reps.emplace_back(
          atom,
          a.function(exact ? Kind::CleanDiv : Kind::FloorDiv, {base, divisor}));
    }
  }
  if (!trunc_reps.empty()) {
    e = a.xreplace(e, trunc_reps);
  }
  // The two FloorDiv passes only rewrite FloorDivs whose Mod is in divisible,
  // which is empty.
  return e;
}

const Expr* NativeShapeEnv::maybe_evaluate_static(const Expr* e) {
  ExprArena& a = *arena_;
  e = a.canonicalize_bool_expr(simplify(e));
  // Substituting the (empty) axioms is the identity.
  auto fs = a.free_symbols(e);
  if (fs.empty() && (e->is_number() || e->is_boolean())) {
    return e;
  }
  std::sort(fs.begin(), fs.end(), [&](const Expr* x, const Expr* y) {
    return a.symbol_info(x).name < a.symbol_info(y).name;
  });
  for (size_t i = 1; i < fs.size(); ++i) {
    if (a.symbol_info(fs[i - 1]).name == a.symbol_info(fs[i]).name) {
      throw NativeUnsupported("sorted(fs, key=str) depends on set order");
    }
  }

  // _maybe_evaluate_static_worker(e, symbol_info, False, False).
  c10::SmallVector<std::pair<const Expr*, const Expr*>, 4> new_shape_env;
  RangeMap new_range_env;
  for (size_t idx = 0; idx < fs.size(); ++idx) {
    const Expr* k = fs[idx];
    auto it = var_to_range_.find(k);
    if (it == var_to_range_.end()) {
      throw NativeUnsupported("symbol without a mirrored range");
    }
    const ValueRanges& vr = it->second;
    if (vr.lower == a.neg_int_oo()) {
      new_range_env.insert_or_assign(k, vr);
      continue;
    }
    if (vr.lower->kind != Kind::Integer) {
      throw NativeUnsupported("int_oo lower bound");
    }
    Facts facts;
    facts.fill(Tri::Unknown);
    facts[static_cast<size_t>(Fact::commutative)] = Tri::True;
    facts[static_cast<size_t>(Fact::positive)] = Tri::True;
    facts[static_cast<size_t>(Fact::integer)] = Tri::True;
    const Expr* s =
        a.symbol("evaluate_static_shape_" + std::to_string(idx), facts);
    const Expr* offset = a.sub(vr.lower, a.integer(1));
    new_shape_env.emplace_back(k, a.add({s, offset}));
    const Expr* upper =
        is_int_oo(vr.upper) ? vr.upper : a.sub(vr.upper, offset);
    new_range_env.insert_or_assign(s, ValueRanges(a.integer(1), upper));
  }
  const Expr* new_expr =
      a.canonicalize_bool_expr(a.safe_expand(a.xreplace(e, new_shape_env)));
  if (new_expr->is_number()) {
    return new_expr;
  }
  ValueRanges out = bound_sympy(a, new_expr, new_range_env);
  return out.is_singleton() ? out.lower : nullptr;
}

const Expr* NativeShapeEnv::maybe_fast_eval_comparison(const Expr* e) {
  const Expr* zero = arena_->integer(0);
  const Expr* sum = nullptr;
  if (e->kind == Kind::Ge && e->args[1] == zero) {
    sum = e->args[0];
  } else if (e->kind == Kind::Le && e->args[0] == zero) {
    sum = e->args[1];
  } else {
    return nullptr;
  }
  // _is_nonneg_sum.
  auto nonneg_term = [&](const Expr* term) {
    if (term->kind == Kind::Symbol) {
      auto it = var_to_range_.find(term);
      return it != var_to_range_.end() &&
          compare_numbers(it->second.lower, zero) >= 0;
    }
    return term->is_number() && compare_numbers(term, zero) >= 0;
  };
  bool nonneg = sum->kind == Kind::Add
      ? std::all_of(sum->args.begin(), sum->args.end(), nonneg_term)
      : nonneg_term(sum);
  return nonneg ? arena_->boolean(true) : nullptr;
}

std::optional<const Expr*> NativeShapeEnv::static_eval(const Expr* e) {
  if (!pristine_ || !all_symbols_mirrored(e)) {
    return std::nullopt;
  }
  const Expr* r = nullptr;
  try {
    r = maybe_evaluate_static(e);
  } catch (const NativeUnsupported&) {
    return std::nullopt;
  }
  log_query({e, false, std::monostate{}, std::nullopt, false}, r);
  return r;
}

std::optional<const Expr*> NativeShapeEnv::evaluate_expr(
    const Expr* e,
    const Hint& hint,
    std::optional<bool> fallback_value) {
  auto r = evaluate_expr_impl(e, hint);
  if (r) {
    log_query({e, true, hint, fallback_value, suppress_guards_tls()}, *r);
  }
  return r;
}

std::optional<const Expr*> NativeShapeEnv::evaluate_expr_impl(
    const Expr* e,
    const Hint& hint) {
  // _evaluate_expr up to the static evaluation; the rest adds a guard or
  // raises.
  if (e->kind == Kind::BooleanTrue || e->kind == Kind::BooleanFalse) {
    return e;
  }
  // Python evaluates the replaced expr.
  if (!replacements_empty_ || !all_symbols_mirrored(e)) {
    return std::nullopt;
  }
  if (e->is_number()) {
    // Python asserts that e equals the hint (Integer(1) == True).
    if (std::holds_alternative<std::monostate>(hint)) {
      return e;
    }
    const auto* i = std::get_if<int64_t>(&hint);
    if (e->kind == Kind::Integer &&
        e->p == (i ? *i : int64_t(std::get<bool>(hint)))) {
      return e;
    }
    return std::nullopt;
  }
  // The range mirror is not updated once the env stops being pristine.
  if (!pristine_) {
    return std::nullopt;
  }
  try {
    if (const Expr* r = maybe_fast_eval_comparison(e)) {
      return r;
    }
    if (const Expr* r = maybe_evaluate_static(e)) {
      return r;
    }
  } catch (const NativeUnsupported&) {
  }
  return std::nullopt;
}

} // namespace torch::symbolic
