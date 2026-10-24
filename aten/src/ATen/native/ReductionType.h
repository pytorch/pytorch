#pragma once

#if !defined(__METAL_VERSION__)
#include <c10/core/Scalar.h>
#endif

namespace at::native {

enum class ReductionType {MAX, MEAN, MIN, SUM, PROD};

#if !defined(__METAL_VERSION__)
inline ReductionType get_reduction_enum(const std::string_view& reduce) {
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
  } else {
    TORCH_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin, got ", reduce);
  }
}

// used for `scatter_reduce`, old options for BC.
inline ReductionType get_operator_enum(const std::string_view reduce, bool use_new_options) {
  if (use_new_options) {
    return get_reduction_enum(reduce);
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
#endif

} // at::native
