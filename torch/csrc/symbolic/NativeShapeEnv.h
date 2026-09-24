#pragma once

#include <torch/csrc/symbolic/ValueRanges.h>

#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// The part of a Python ShapeEnv that native evaluation reads, mirrored by the
// Python side (Python stays authoritative), and ports of the ShapeEnv
// evaluation steps. Static evaluation answers only while the env is pristine:
// no guards, deferred runtime asserts, axioms, divisible, replacements or
// post-creation range updates, so Python's version-keyed caches cannot hold
// an answer computed in a different state.

namespace torch::symbolic {

// None, int or bool, as passed to ShapeEnv.evaluate_expr.
using Hint = std::variant<std::monostate, int64_t, bool>;

// A query answered natively. Python replays it (_flush_native_queries) before
// it next evaluates or mutates any ShapeEnv, so that its caches, which do not
// key on guards, axioms or ranges and are shared by all envs, match a run
// without native answers.
struct NativeQuery {
  const Expr* expr;
  // _maybe_evaluate_static(expr) if false, else evaluate_expr(expr, hint,
  // fx_node=False, size_oblivious=False, fallback_value) under the
  // suppress_guards TLS.
  bool evaluate;
  Hint hint;
  std::optional<bool> fallback_value;
  bool suppress_guards;

  bool operator==(const NativeQuery& o) const {
    return expr == o.expr && evaluate == o.evaluate && hint == o.hint &&
        fallback_value == o.fallback_value &&
        suppress_guards == o.suppress_guards;
  }
};

struct LoggedQuery {
  // Position in the last-use order over all envs.
  uint64_t seq;
  NativeQuery query;
  // nullptr for None.
  const Expr* result;
};

// Mirror of the Python ShapeEnv suppress_guards TLS.
bool& suppress_guards_tls();

class NativeShapeEnv : public c10::intrusive_ptr_target {
 public:
  explicit NativeShapeEnv(c10::intrusive_ptr<ExprArena> arena)
      : arena_(std::move(arena)) {}
  ~NativeShapeEnv() override;

  ExprArena& arena() {
    return *arena_;
  }

  void add_symbol(
      const Expr* sym,
      std::optional<int64_t> hint,
      const ValueRanges& range,
      bool size_like);
  void update_range(const Expr* sym, const ValueRanges& range);
  void mark_not_pristine() {
    pristine_ = false;
  }
  void mark_replacements() {
    replacements_empty_ = false;
    pristine_ = false;
  }
  bool pristine() const {
    return pristine_;
  }
  bool replacements_empty() const {
    return replacements_empty_;
  }
  // (hint, range, size_like) of a mirrored symbol.
  std::optional<std::tuple<std::optional<int64_t>, ValueRanges, bool>> mirrored(
      const Expr* sym) const;

  // ShapeEnv.simplify(e), ShapeEnv._maybe_evaluate_static(e) (nullptr for
  // None) and ShapeEnv._maybe_fast_eval_comparison(e) (nullptr for None), with
  // default arguments. The first two require a pristine env and every free
  // symbol mirrored.
  const Expr* simplify(const Expr* e);
  const Expr* maybe_evaluate_static(const Expr* e);
  const Expr* maybe_fast_eval_comparison(const Expr* e);

  // Entry points; nullopt means the caller must delegate to Python.
  // Answers are logged for replay.
  // _maybe_evaluate_static(e) as _static_eval_sym_bool calls it.
  std::optional<const Expr*> static_eval(const Expr* e);
  // ShapeEnv.evaluate_expr(e, hint, fallback_value=fallback_value) where it
  // returns without adding a guard. The caller checks that
  // aggressive_guard_free_semantics and backed_size_oblivious are unset.
  std::optional<const Expr*> evaluate_expr(
      const Expr* e,
      const Hint& hint,
      std::optional<bool> fallback_value = std::nullopt);

  // Distinct answered queries since the last call, in last-use order.
  std::vector<LoggedQuery> take_queries();
  // Whether any env has queries to replay.
  static bool queries_pending();

 private:
  struct QueryHash {
    size_t operator()(const NativeQuery& q) const;
  };

  std::optional<const Expr*> evaluate_expr_impl(const Expr* e, const Hint& hint);
  void log_query(const NativeQuery& q, const Expr* result);

  using LeCache = std::map<std::tuple<const Expr*, const Expr*, bool>, bool>;

  bool all_symbols_mirrored(const Expr* e) const;
  std::optional<const Expr*> lower_bound(const Expr* e);
  bool is_nonneg_term(const Expr* term);
  bool definitely_le(
      const Expr* a,
      const Expr* b,
      bool use_static_fallback,
      LeCache& le_cache);

  c10::intrusive_ptr<ExprArena> arena_;
  RangeMap var_to_range_;
  std::unordered_map<const Expr*, int64_t> backed_var_to_val_;
  std::unordered_set<const Expr*> size_like_;
  bool pristine_ = true;
  bool replacements_empty_ = true;
  std::unordered_map<NativeQuery, std::pair<uint64_t, const Expr*>, QueryHash>
      queries_;
};

} // namespace torch::symbolic
