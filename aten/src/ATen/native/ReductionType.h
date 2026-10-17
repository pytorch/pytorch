#pragma once

#include <c10/core/Scalar.h>

namespace at::native {

enum class ReductionType {MAX, MEAN, MIN, SUM, PROD, NONE};

// `allow_none` is opt-in: ReductionType::NONE is only implemented for
// scatter_reduce. index_reduce, segment_reduce and sparse mm_reduce share this
// parser and would silently fall through to their default branch.
inline ReductionType get_reduction_enum(const std::string_view& reduce, bool allow_none = false) {
  if (reduce == "max" || reduce == "amax") {
    return ReductionType::MAX;
  } else if (reduce == "mean") {
    return ReductionType::MEAN;
  } else if (reduce == "min" || reduce == "amin") {
    return ReductionType::MIN;
  } else if (reduce == "sum") {
    return ReductionType::SUM;
  } else if (reduce == "prod") {
    return ReductionType::PROD;
  } else if (allow_none && (reduce == "none" || reduce == "last")) {
    return ReductionType::NONE;
  } else {
    TORCH_CHECK(false,
                allow_none
                    ? "reduce argument must be either sum, prod, mean, amax, amin, none or last, got "
                    : "reduce argument must be either sum, prod, mean, amax or amin, got ",
                reduce);
  }
}

// used for `scatter_reduce`, old options for BC.
inline ReductionType get_operator_enum(const std::string_view reduce, bool use_new_options, bool allow_none = false) {
  if (use_new_options) {
    return get_reduction_enum(reduce, allow_none);
  } else {
    if (reduce == "add") {
      return ReductionType::SUM;
    } else if (reduce == "multiply") {
      return ReductionType::PROD;
    } else {
      TORCH_CHECK(false, "reduce argument must be either add or multiply.")
    }
  }
}

} // at::native
