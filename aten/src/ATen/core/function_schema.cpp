#include <ATen/core/function_schema.h>

#include <iostream>
#include <stack>
#include <utility>

namespace c10 {

void FunctionSchema::dump() const {
  std::cout << *this << "\n";
}

const std::vector<Argument>& FunctionSchema::getCorrectList(SchemaArgType type) const {
  if (type == SchemaArgType::input) {
    return arguments();
  } else {
    return returns();
  }
}

FunctionSchema FunctionSchema::cloneWithRealTypes(bool with_symint) const {
  auto alwaysCloneWithRealTypes = [&](const Argument& a) {
    return a.cloneWithType(a.real_type());
  };
  auto cloneWithRealTypes = [&](const Argument& a) {
    if (with_symint) {
      return a.cloneWithType(a.real_type());
    }
    // Don't use real type if it looks like a SymInt
    // NB: keep this in sync with unpackSymInt in KernelFunction_impl.h
    if (
      *a.real_type() == *getTypePtr<c10::SymInt>() ||
      *a.real_type() == *getTypePtr<std::optional<c10::SymInt>>() ||
      *a.real_type() == *getTypePtr<c10::SymIntArrayRef>() ||
      *a.real_type() == *getTypePtr<at::OptionalSymIntArrayRef>()
    ) {
      // Keep the fake type
      return a.cloneWithType(a.type());
    } else {
      return a.cloneWithType(a.real_type());
    }
  };
  std::vector<Argument> new_arguments, new_returns;
  std::transform(arguments().begin(), arguments().end(), std::back_inserter(new_arguments), cloneWithRealTypes);
  // NB: SymInt returns are always SymInt
  std::transform(returns().begin(), returns().end(), std::back_inserter(new_returns), alwaysCloneWithRealTypes);
  return FunctionSchema(
    name(),
    overload_name(),
    std::move(new_arguments),
    std::move(new_returns),
    is_vararg(),
    is_varret());
}

bool FunctionSchema::canAliasTypeSetsAlias(const std::optional<AliasTypeSet> &lhs, const std::optional<AliasTypeSet> &rhs) const {
  if (!lhs || !rhs) {
    return false;
  }
  for (const TypePtr& lhsType : *lhs) {
    for (const TypePtr& rhsType : *rhs) {
      if (lhsType == rhsType) {
        return true;
      }
    }
  }
  return false;
}

std::optional<AliasTypeSet> FunctionSchema::getAliasTypeSetContainedTypes(const std::optional<AliasTypeSet> &aliasTypeSet) const {
  if (!aliasTypeSet) {
    return c10::nullopt;
  }
  std::unordered_set<TypePtr> containedTypes;
  std::stack<TypePtr> typeStack;
  // Push all 1st level contained types into the stack.
  for (const TypePtr& type: *aliasTypeSet) {
    for (const TypePtr& containedType : type->containedTypes()){
      typeStack.push(containedType);
    }
  }

  // process all further level contained types.
  while (!typeStack.empty()) {
    TypePtr current = typeStack.top();
    typeStack.pop();
    if (!containedTypes.count(current)) {
      for (const TypePtr& containedType : current->containedTypes()) {
        typeStack.push(containedType);
      }
    }
    containedTypes.insert(current);
  }

  return AliasTypeSet(containedTypes.begin(), containedTypes.end());
}

std::optional<AliasTypeSet> FunctionSchema::mapTypeToAliasTypeSet(const TypePtr& type) const {
  switch(type->kind()) {
    case TypeKind::ListType:
    case TypeKind::DictType:
    case TypeKind::ClassType:
    case TypeKind::TensorType:
      return AliasTypeSet {c10::unshapedType(type)};
    case TypeKind::UnionType: {
      AliasTypeSet mutable_types;
      for (const TypePtr& inner :
            type->expectRef<UnionType>().containedTypes()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      if (mutable_types.empty()) {
        return c10::nullopt;
      }
      return mutable_types;
    }
    case TypeKind::AnyType:
      return {AliasTypeSet{type}};
    case TypeKind::OptionalType: {
      auto inner = type->castRaw<OptionalType>()->getElementType();
      return mapTypeToAliasTypeSet(inner);
    }
    case TypeKind::TupleType: {
      AliasTypeSet mutable_types;
      for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      if (mutable_types.empty()) {
        return c10::nullopt;
      }
      return {AliasTypeSet{TupleType::create(std::move(mutable_types))}};
    }
    default:
      return c10::nullopt;
  }
}

bool FunctionSchema::may_alias(const SchemaArgument& lhs, const SchemaArgument& rhs) const {
  TORCH_INTERNAL_ASSERT(
      (lhs.index < getCorrectList(lhs.type).size()),
      "Invalid index for schema.");
  TORCH_INTERNAL_ASSERT(
      (rhs.index < getCorrectList(rhs.type).size()),
      "Invalid index for schema.");

  const Argument lhsArg = getCorrectList(lhs.type)[lhs.index];
  const Argument rhsArg = getCorrectList(rhs.type)[rhs.index];

  std::optional<AliasTypeSet> lhsTypes = mapTypeToAliasTypeSet(lhsArg.type());
  std::optional<AliasTypeSet> rhsTypes = mapTypeToAliasTypeSet(rhsArg.type());

  // Check to see if lhs and rhs have the same alias set
  if (canAliasTypeSetsAlias(lhsTypes, rhsTypes)) {
    if (lhsArg.alias_info() && rhsArg.alias_info()) {
      for (const auto& lhsSet : lhsArg.alias_info()->afterSets()) {
        for (const auto& rhsSet : rhsArg.alias_info()->afterSets()) {
          if (lhsSet == rhsSet) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool FunctionSchema::may_contain_alias(const SchemaArgument& lhs, const SchemaArgument& rhs, bool bidirectional) const {
  bool may_alias_result = may_alias(lhs, rhs);
  if (may_alias_result) {
    return true;
  }

  const c10::Argument lhsArg = getCorrectList(lhs.type)[lhs.index];
  const c10::Argument rhsArg = getCorrectList(rhs.type)[rhs.index];
  std::optional<AliasTypeSet> lhsTypes = mapTypeToAliasTypeSet(lhsArg.type());
  std::optional<AliasTypeSet> rhsTypes = mapTypeToAliasTypeSet(rhsArg.type());
  std::optional<AliasTypeSet> lhsContainedTypes = getAliasTypeSetContainedTypes(lhsTypes);
  std::optional<AliasTypeSet> rhsContainedTypes = getAliasTypeSetContainedTypes(rhsTypes);

  // Checks if one side is wildcard and the other side is a container of the same type
  bool lhsWildcard = lhsArg.alias_info() && lhsArg.alias_info()->isWildcardAfter() && canAliasTypeSetsAlias(lhsTypes, rhsContainedTypes);
  bool rhsWildcard = rhsArg.alias_info() && rhsArg.alias_info()->isWildcardAfter() && canAliasTypeSetsAlias(rhsTypes, lhsContainedTypes);

  if (bidirectional) {
    return lhsWildcard || rhsWildcard || canAliasTypeSetsAlias(lhsContainedTypes, rhsContainedTypes);
  } else {
    return rhsWildcard || canAliasTypeSetsAlias(lhsContainedTypes, rhsContainedTypes);
  }
}
} // namespace c10
