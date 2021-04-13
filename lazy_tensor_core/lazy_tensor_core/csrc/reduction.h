#pragma once

#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

}  // namespace torch_lazy_tensors
