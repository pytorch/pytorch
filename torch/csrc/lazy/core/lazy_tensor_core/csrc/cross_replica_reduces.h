#pragma once

#include <vector>

#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {

enum class AllReduceType {
  kSum,
  kMin,
  kMax,
  kMul,
  kOr,
  kAnd,
};

}  // namespace torch_lazy_tensors
