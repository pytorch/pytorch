#pragma once

#include <ATen/core/operator_name.h>
#include <string>
#include <unordered_set>

namespace c10 {

struct TORCH_API ObservedOperators {
  ObservedOperators() = delete;

  static bool isObserved(const OperatorName& name);

  static std::unordered_set<std::string>& getUnobservedOperatorList();
};

} // namespace c10
