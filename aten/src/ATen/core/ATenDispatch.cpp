#include <ATen/core/ATenDispatch.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

std::unordered_set<c10::OperatorName> aten_ops_already_moved_to_c10_0();
std::unordered_set<c10::OperatorName> aten_ops_already_moved_to_c10_1();
std::unordered_set<c10::OperatorName> aten_ops_already_moved_to_c10_2();

namespace {
std::unordered_set<c10::OperatorName> aten_ops_already_moved_to_c10_() {
  std::unordered_set<c10::OperatorName> result;
  for (c10::OperatorName name : aten_ops_already_moved_to_c10_0()) {
    result.insert(std::move(name));
  }
  for (c10::OperatorName name : aten_ops_already_moved_to_c10_1()) {
    result.insert(std::move(name));
  }
  for (c10::OperatorName name : aten_ops_already_moved_to_c10_2()) {
    result.insert(std::move(name));
  }
  return result;
}
}

const std::unordered_set<c10::OperatorName>& aten_ops_already_moved_to_c10() {
  static auto ops = aten_ops_already_moved_to_c10_();
  return ops;
}

} // namespace at
