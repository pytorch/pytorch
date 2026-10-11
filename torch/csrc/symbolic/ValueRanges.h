#pragma once

#include <torch/csrc/symbolic/Expr.h>

#include <unordered_map>

// Port of torch/utils/_sympy/value_ranges.py.

namespace torch::symbolic {

// A ValueRanges whose bounds are numbers or BooleanTrue/BooleanFalse.
struct ValueRanges {
  // ValueRanges.__init__ for bounds that are already normalized: throws for an
  // invalid range or one that make_value_range would change.
  ValueRanges(const Expr* lower, const Expr* upper);

  static bool is_int_bound(const Expr* e) {
    return e->kind == Kind::Integer || e->kind == Kind::IntInfinity ||
        e->kind == Kind::NegativeIntInfinity;
  }

  bool is_bool() const {
    return lower->is_boolean();
  }
  bool is_int() const {
    return is_int_bound(lower) && is_int_bound(upper);
  }
  bool is_singleton() const {
    return lower == upper;
  }

  const Expr* lower;
  const Expr* upper;
};

// ValueRanges.__init__: [Integer, oo] becomes [Integer, int_oo] and
// [-oo, Integer] becomes [-int_oo, Integer].
ValueRanges make_value_range(
    ExprArena& arena,
    const Expr* lower,
    const Expr* upper);

using RangeMap = std::unordered_map<const Expr*, ValueRanges>;

// sympy_interp(SymPyValueRangeAnalysis, ranges, e,
//              missing_handler=_default_symbol_range)
ValueRanges value_range_interp(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges);

// bound_sympy(e, {**context_ranges, **ranges}), where context_ranges is the
// TracingContext's shape_env.var_to_range.
ValueRanges bound_sympy(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges,
    const RangeMap* context_ranges = nullptr);

} // namespace torch::symbolic
