#include <ATen/core/OpsAlreadyMovedToC10.h>

// ${generated_comment}

namespace at {

// TODO Once all ATen ops are moved to c10, this file should be removed

const std::unordered_set<c10::OperatorName>& aten_ops_already_moved_to_c10() {
  static std::unordered_set<c10::OperatorName> result {
    ${c10_ops_already_moved_from_aten_to_c10}
    {"", ""}
  };
  return result;
}

const std::unordered_set<c10::OperatorName>& aten_ops_not_moved_to_c10_yet() {
  static std::unordered_set<c10::OperatorName> result {
    ${c10_ops_not_moved_from_aten_to_c10_yet}
    {"", ""}
  };
  return result;
}

}
