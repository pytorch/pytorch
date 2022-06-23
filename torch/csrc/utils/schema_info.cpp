#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {

bool SchemaInfo::isMutating(int index) const {
  TORCH_INTERNAL_ASSERT(
      index < schema_.arguments().size() && index >= 0,
      "Invalid index for schema.");

  return schema_.arguments()[index].alias_info() != nullptr &&
      schema_.arguments()[index].alias_info()->isWrite();
}

bool SchemaInfo::isMutating(c10::string_view name) const {
  c10::optional<int> index = schema_.argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);

  return isMutating(*index);
}

bool SchemaInfo::areAliasing(
    const SchemaArgument& lhs,
    const SchemaArgument& rhs) const {
  TORCH_INTERNAL_ASSERT(
      (lhs.type == input && lhs.index < schema_.arguments().size() &&
       lhs.index >= 0) ||
          (lhs.type == output && lhs.index < schema_.returns().size() &&
           lhs.index >= 0),
      "Invalid index for schema.");
  TORCH_INTERNAL_ASSERT(
      (rhs.type == input && rhs.index < schema_.arguments().size() &&
       rhs.index >= 0) ||
          (rhs.type == output && rhs.index < schema_.returns().size() &&
           rhs.index >= 0),
      "Invalid index for schema.");

  const c10::AliasInfo* lhsAliasInfo =
      schema_.arguments()[lhs.index].alias_info();
  const c10::AliasInfo* rhsAliasInfo =
      schema_.arguments()[rhs.index].alias_info();

  if ((lhsAliasInfo && lhsAliasInfo->isWildcardAfter()) ||
      (rhsAliasInfo && rhsAliasInfo->isWildcardAfter())) {
    return true;
  }

  if (lhsAliasInfo && rhsAliasInfo) {
    for (const auto& lhsSet : lhsAliasInfo->afterSets()) {
      for (const auto& rhsSet : rhsAliasInfo->afterSets()) {
        if (lhsSet == rhsSet) {
          return true;
        }
      }
    }
  }
  return false;
}
} // namespace utils
} // namespace torch
