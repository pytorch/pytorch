#pragma once

#include <torch/csrc/symbolic/Expr.h>

#include <unordered_map>

// Port of torch/utils/_sympy/value_ranges.py for integer and boolean ranges.

namespace torch::symbolic {

// A ValueRanges whose bounds are Integers or +-int_oo (is_int) or
// BooleanTrue/BooleanFalse (is_bool). Float ranges are not ported: building
// one throws NativeUnsupported.
struct ValueRanges {
  // ValueRanges.__init__: throws for an invalid or float range.
  ValueRanges(const Expr* lower, const Expr* upper);

  bool is_bool() const {
    return lower->is_boolean();
  }
  bool is_singleton() const {
    return lower == upper;
  }

  const Expr* lower;
  const Expr* upper;
};

using RangeMap = std::unordered_map<const Expr*, ValueRanges>;

// sympy_interp(SymPyValueRangeAnalysis, ranges, e,
//              missing_handler=_default_symbol_range)
ValueRanges value_range_interp(
    ExprArena& arena,
    const Expr* e,
    const RangeMap& ranges);

} // namespace torch::symbolic
