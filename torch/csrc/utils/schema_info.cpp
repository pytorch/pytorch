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

  const c10::Argument lhsArg = schema_.arguments()[lhs.index];
  const c10::Argument rhsArg = schema_.arguments()[rhs.index];

  if (lhsArg.alias_info() && lhsArg.alias_info()->isWildcardAfter() &&
      rhsArg.alias_info() && rhsArg.alias_info()->isWildcardAfter()) {
    if (lhsArg.type()->kind() == rhsArg.type()->kind()) {
      return true;
    } else {
      for (const auto& type : lhsArg.type()->containedTypes()) {
        if (type->kind() == rhsArg.type()->kind()) {
          return true;
        }
      }
      for (const auto& type : rhsArg.type()->containedTypes()) {
        if (type->kind() == lhsArg.type()->kind()) {
          return true;
        }
      }
    }
  }

  if (lhsArg.alias_info() && rhsArg.alias_info()) {
    for (const auto& lhsSet : lhsArg.alias_info()->afterSets()) {
      for (const auto& rhsSet : rhsArg.alias_info()->afterSets()) {
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
