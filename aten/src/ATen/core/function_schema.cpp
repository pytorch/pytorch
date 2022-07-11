#include <ATen/core/function_schema.h>

#include <iostream>

namespace c10 {

void FunctionSchema::dump() const {
  std::cout << *this << "\n";
}

std::vector<Argument> FunctionSchema::getCorrectList(SchemaArgType type) const {
  if (type == SchemaArgType::input) {
    return arguments();
  } else {
    return returns();
  }
}

c10::optional<std::vector<TypePtr>> FunctionSchema::mapTypeToAliasTypeSet(const TypePtr& type) const {
  switch(type->kind()) {
    case TypeKind::ListType:
    case TypeKind::DictType:
    case TypeKind::ClassType:
    case TypeKind::TensorType:
      return std::vector<TypePtr> {c10::unshapedType(type)};
    case TypeKind::UnionType: {
      std::vector<TypePtr> mutable_types;
      for (const TypePtr& inner :
            type->expectRef<UnionType>().containedTypes()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      if (mutable_types.size() == 0) {
        return c10::nullopt;
      }
      return mutable_types;
    }
    case TypeKind::AnyType:
      return {std::vector<TypePtr>{type}};
    case TypeKind::OptionalType: {
      auto inner = type->castRaw<OptionalType>()->getElementType();
      return mapTypeToAliasTypeSet(inner);
    }
    case TypeKind::TupleType: {
      std::vector<TypePtr> mutable_types;
      for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      if (mutable_types.size() == 0) {
        return c10::nullopt;
      }
      return {std::vector<TypePtr>{TupleType::create(mutable_types)}};
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

  c10::optional<std::vector<TypePtr>> lhsTypes = mapTypeToAliasTypeSet(lhsArg.type());
  c10::optional<std::vector<TypePtr>> rhsTypes = mapTypeToAliasTypeSet(rhsArg.type());

  // Check to see if the lhs and rhs types can alias each other
  bool typesCanAlias = false;
  if (lhsTypes && rhsTypes) {
    for (const TypePtr& lhsType : *lhsTypes) {
      for (const TypePtr& rhsType : *rhsTypes) {
        if (lhsType == rhsType) {
          typesCanAlias = true;
        }
      }
    }
  }

  // Check to see if lhs and rhs have the same alias set
  if (typesCanAlias) {
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

}
