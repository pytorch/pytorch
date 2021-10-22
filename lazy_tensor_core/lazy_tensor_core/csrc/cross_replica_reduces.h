#pragma once

#include <vector>


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
