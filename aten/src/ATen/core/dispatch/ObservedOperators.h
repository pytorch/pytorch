#pragma once

#include <ATen/core/operator_name.h>

namespace c10 {

struct TORCH_API ObservedOperators {
  ObservedOperators() = delete;

  static bool isObserved(const OperatorName& name);
};

} // namespace c10
