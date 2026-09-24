#pragma once

#include <torch/csrc/symbolic/ValueRanges.h>

#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

// The part of a Python ShapeEnv that native evaluation reads, mirrored by the
// Python side (Python stays authoritative), and ports of the ShapeEnv
// evaluation steps. Static evaluation answers only while the env is pristine:
// no guards, deferred runtime asserts, axioms, divisible, replacements or
// post-creation range updates, so Python's version-keyed caches cannot hold
// an answer computed in a different state.

namespace torch::symbolic {

class NativeShapeEnv : public c10::intrusive_ptr_target {
 public:
  explicit NativeShapeEnv(c10::intrusive_ptr<ExprArena> arena)
      : arena_(std::move(arena)) {}

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

  // ShapeEnv.simplify(e), ShapeEnv._maybe_evaluate_static(e) (nullptr for
  // None) and ShapeEnv._maybe_fast_eval_comparison(e) (nullptr for None), with
  // default arguments. The first two require a pristine env and every free
  // symbol mirrored.
  const Expr* simplify(const Expr* e);
  const Expr* maybe_evaluate_static(const Expr* e);
  const Expr* maybe_fast_eval_comparison(const Expr* e);

  // Entry points; nullopt means the caller must delegate to Python.
  // _maybe_evaluate_static(e) as _static_eval_sym_bool calls it.
  std::optional<const Expr*> static_eval(const Expr* e);
  // ShapeEnv.evaluate_expr(e, hint) where it returns without adding a guard.
  // hint is None or an int.
  std::optional<const Expr*> evaluate_expr(
      const Expr* e,
      std::optional<int64_t> hint);

 private:
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
};

} // namespace torch::symbolic
