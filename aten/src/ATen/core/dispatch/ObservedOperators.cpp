#include <ATen/core/dispatch/ObservedOperators.h>

#include <string>
#include <unordered_set>

namespace c10 {

/* static */
std::unordered_set<std::string>& ObservedOperators::getUnobservedOperatorList() {
  // names of the operators that should not be observed
  static std::unordered_set<std::string> not_observed_ops = {
    "aten::size",
    "aten::is_leaf",
    "aten::output_nr",
    "aten::_version",
    "aten::is_complex",
    "profiler::_record_function_enter",
    "profiler::_record_function_enter_new",
    "profiler::_record_function_exit",
  };
  return not_observed_ops;
}

/* static */
bool ObservedOperators::isObserved(const OperatorName& name) {
  return !ObservedOperators::getUnobservedOperatorList().count(name.name);
}

} // namespace c10
