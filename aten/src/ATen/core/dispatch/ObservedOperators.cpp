#include <ATen/core/dispatch/ObservedOperators.h>

#include <string>
#include <unordered_set>

namespace c10 {

/* static */
bool ObservedOperators::isObserved(const OperatorName& name) {
  // names of the operators that should not be observed
  std::unordered_set<std::string> not_observed_ops = {
    "aten::size",
    "aten::is_leaf",
    "aten::output_nr",
    "aten::_version",
    "aten::is_complex",
    "profiler::_record_function_enter",
    "profiler::_record_function_exit",
  };
  return !not_observed_ops.count(name.name);
}

} // namespace c10
