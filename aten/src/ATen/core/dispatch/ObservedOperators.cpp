#include <ATen/core/dispatch/ObservedOperators.h>

#include <string>
#include <unordered_set>

namespace c10 {

/* static */
bool ObservedOperators::isObserved(const OperatorName& name) {
  // names of the operators that should not be observed
  std::unordered_set<std::string> not_observed_ops = {
    "aten::size",
  };
  return !not_observed_ops.count(toString(name));
}

} // namespace c10
