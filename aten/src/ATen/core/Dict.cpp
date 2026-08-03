#include <ATen/core/Dict.h>


namespace c10::detail {
bool operator==(const DictImpl& lhs, const DictImpl& rhs) {
  bool isEqualFastChecks =
      *lhs.elementTypes.keyType == *rhs.elementTypes.keyType &&
      *lhs.elementTypes.valueType == *rhs.elementTypes.valueType &&
      lhs.dict.size() == rhs.dict.size();
  if (!isEqualFastChecks) {
    return false;
  }

  // Dict equality should not care about ordering.
  for (const auto& pr : lhs.dict) {
    auto it = rhs.dict.find(pr.first);
    if (it == rhs.dict.cend()) {
      return false;
    }
    // see: [container equality]
    if (!_fastEqualsForContainer(it->second, pr.second)) {
      return false;
    }
  }

  return true;
}
} // namespace c10::detail
