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
    return std::nullopt;
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
        return std::nullopt;
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
        return std::nullopt;
      }
      return {AliasTypeSet{TupleType::create(std::move(mutable_types))}};
    }
    default:
      return std::nullopt;
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

std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  out << schema.name();
  if (!schema.overload_name().empty()) {
    out << "." << schema.overload_name();
  }
  out << "(";

  bool seen_kwarg_only = false;
  for (const auto i : c10::irange(schema.arguments().size())) {
    if (i > 0) out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    out << schema.arguments()[i];
  }

  if(schema.is_vararg()) {
    if(!schema.arguments().empty())
      out << ", ";
    out << "...";
  }

  out << ") -> ";

  const auto& returns = schema.returns();

  /*
   * We should skip parenthesis if we return a single item and it's not varret,
   * or we return nothing but varret.
   *
   * Need special handling for schema
   *   aten::items.str(Dict(str, t) self) -> (str,t)[]
   * Even though this schema returns a single item, we need add parenthesis.
   * The is necessary so the printed schema can be parsed by the C++ SchemaParser
   * Without the extra parenthesis, the parser sees the first parenthesis in '(str,t)' and mistakenly
   * treat the return type as a tuple. An alternative is to enhance the Lexer
   * to lookahead multiple tokens to accurately decide if the return type is
   * a tuple.
   */
  bool need_paren = !(
    (returns.size() == 1 && !schema.is_varret()) ||
    (returns.empty() && schema.is_varret()));

  if (returns.size() == 1 && !schema.is_varret()) {
    std::stringstream return_ss;
    return_ss << returns.at(0);
    auto return_str = return_ss.str();

    // enclosing the single return item with parenthesis if the return type
    // starts with a left parenthesis.
    //
    // There are 2 cases
    // 1. something like 'aten::items.str(Dict(str, t) self) -> ((str, t)[])'.
    // without the extra parenthesis, the c++ schem parser can not parse it.
    // 2. something like '-> ((str, str))'. Need extra parenthesis so the return
    // type is a single tuple rather than two strings.
    // PR (https://github.com/pytorch/pytorch/pull/23204) has more context about
    // this. test_serialize_and_deserialize (https://github.com/pytorch/pytorch/blob/master/test/test_function_schema.py#L15)
    // also covers this case.
    if (!return_str.empty() && return_str.front() == '(') {
      need_paren = true;
    }
  }

  if (need_paren) {
    out << "(";
  }
  for (const auto i : c10::irange(returns.size())) {
    if (i > 0) {
      out << ", ";
    }
    out << returns.at(i);
  }
  if (schema.is_varret()) {
    if (!returns.empty()) {
      out << ", ";
    }
    out << "...";
  }
  if (need_paren) {
    out << ")";
  }
  return out;
}

static size_t findFirstOutArg(const std::vector<Argument>& args) {
  // find the start of out args in the schema
  for (const auto out_start_idx : c10::irange(args.size())) {
    if (args.at(out_start_idx).is_out()) {
      return out_start_idx;
    }
  }
  return args.size();
}

bool Argument::isBackwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not) const {
    const Argument* lhs = this;
    const Argument* rhs = &old;
    if (!(lhs->name() == rhs->name()
        && lhs->N() == rhs->N()
          && (lhs->alias_info() == rhs->alias_info()
              || (lhs->alias_info() != nullptr && rhs->alias_info() != nullptr
                  && *lhs->alias_info() == *rhs->alias_info())))) {
      return false;
    }
    if (lhs->kwarg_only() && !rhs->kwarg_only()) {
      return false;
    }
    if (!rhs->type()->isSubtypeOfExt(*lhs->type(), why_not)) {
      return false;
    }
    if (rhs->default_value().has_value() &&
        lhs->default_value() != rhs->default_value()) {
      return false;
    }
    return true;
}

bool Argument::isForwardCompatibleWith(
    const Argument& old,
    std::ostream* why_not) const {
  const Argument* lhs = this;
  const Argument* rhs = &old;
  if (!(lhs->name() == rhs->name()
      && lhs->N() == rhs->N()
        && (lhs->alias_info() == rhs->alias_info()
            || (lhs->alias_info() != nullptr && rhs->alias_info() != nullptr
                && *lhs->alias_info() == *rhs->alias_info())))) {
    return false;
  }
  if (lhs->kwarg_only() && !rhs->kwarg_only()) {
    return false;
  }
  if (!lhs->type()->isSubtypeOfExt(rhs->type(), why_not)) {
    return false;
  }
  if (rhs->default_value().has_value() &&
      lhs->default_value() != rhs->default_value()) {
    return false;
  }
  if (lhs->default_value().has_value() && !rhs->default_value().has_value()) {
    return false;
  }
  return true;
}

std::string FunctionSchema::formatTypeMismatchMsg(
    const Argument& expected,
    const std::string& actual_type,
    std::optional<size_t> position,
    std::optional<std::string> value) const {
  std::string position_str;
  if (position) {
    position_str = c10::str("Position: ", *position, "\n");
  }
  std::string value_str;
  if (value) {
    value_str = c10::str("Value: ", *value, "\n");
  }
  return c10::str(
      name(),
      "() ",
      expected.formatTypeMismatchMsg(actual_type),
      position_str,
      value_str,
      "Declaration: ",
      *this);
}

bool FunctionSchema::isBackwardCompatibleWith(
    const FunctionSchema& old,
    std::ostream* why_not) const {
  if (!(name() == old.name()
        && overload_name() == old.overload_name()
        // we are conservative on is_vararg and is_varret,
        // since they are only used by internal operators
        && is_vararg() == old.is_vararg()
        && is_varret() == old.is_varret()
        && returns().size() == old.returns().size()
        && arguments().size() >= old.arguments().size())) {
    return false;
  }
  for (const auto i : c10::irange(returns().size())) {
    // Backwards compatibility requires covariance on argument types
    // (i.e. more generic), and contravariance on return types (i.e.
    //  more specific).
    if (!old.returns().at(i).isBackwardCompatibleWith(
          returns().at(i),
          why_not)) {
      return false;
    }
  }

  // we want to test both out and default args separately
  size_t old_out_start_idx = findFirstOutArg(old.arguments());
  size_t new_out_start_idx = findFirstOutArg(arguments());

  // make sure among the default args, they are backward compatible
  for (const auto i : c10::irange(old_out_start_idx)) {
    if (!arguments().at(i).isBackwardCompatibleWith(
          old.arguments().at(i), why_not)) {
      return false;
    }
  }

  // Validate that all new arguments provided has a default value
  for (const auto i : c10::irange(old_out_start_idx, new_out_start_idx)) {
    if (!arguments().at(i).default_value()) {
      if (why_not) {
        *why_not
            << "Function schema not backward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str()
            << " did not provide a default value.";
      }
      return false;
    }
  }

  // now compare the out args
  for (const auto i : c10::irange(old_out_start_idx, old.arguments().size())) {
    if (!arguments()
             .at(i - old_out_start_idx + new_out_start_idx)
             .isBackwardCompatibleWith(old.arguments().at(i), why_not)) {
      return false;
    }
  }

  return true;
}

bool FunctionSchema::isForwardCompatibleWith(
    const FunctionSchema& old,
    std::ostringstream& why_not) const {
  if (!(name() == old.name() &&
        overload_name() == old.overload_name()
        // we are conservative on is_vararg and is_varret,
        // since they are only used by internal operators
        && is_vararg() == old.is_vararg() && is_varret() == old.is_varret() &&
        returns().size() == old.returns().size())) {
    return false;
  }

  // we want to test both out and default args separately
  size_t old_out_start_idx = findFirstOutArg(old.arguments());
  size_t new_out_start_idx = findFirstOutArg(arguments());

  if (old.arguments().size() - old_out_start_idx !=
      arguments().size() - new_out_start_idx) {
    if (why_not) {
      why_not << "Function schema should have the "
              << "same number of out arguments";
    }
    return false;
  }

  // make sure among the default args, they are forward compatible
  for (size_t i = 0; i < std::min(old_out_start_idx, new_out_start_idx); i++) {
    if (!arguments().at(i).isForwardCompatibleWith(old.arguments().at(i))) {
      if (why_not) {
        why_not
            << "'" << arguments().at(i).name() << "'"
            << " is not forward compatible with the older version of the schema";
      }
      return false;
    }
  }

  // Validate that all new arguments provided has a default value
  for (size_t i = old_out_start_idx; i < new_out_start_idx; ++i) {
    auto const &default_value = arguments().at(i).default_value();
    if (!default_value.has_value()) {
      if (why_not) {
        why_not
            << "Function schema is not forward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str()
            << " did not provide a default value.";
      }
      return false;
    }

    if (default_value->isList() || default_value->isGenericDict()) {
      if (why_not) {
        why_not
            << "Function schema is not forward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str() << " has a container type "
            << "as its default value.";
      }
      return false;
    }
  }

  // now compare the out args
  for (size_t i = old_out_start_idx; i < old.arguments().size(); i++) {
    if (!arguments()
             .at(i - old_out_start_idx + new_out_start_idx)
             .isForwardCompatibleWith(old.arguments().at(i))) {
      if (why_not) {
        why_not << "Out argument '"
                << "'" << arguments().at(i).name()
                << " is not FC with the older version of the schema";
      }
      return false;
    }
  }

  return true;
}

std::string FunctionSchema::findErrorInKwargs(const std::vector<std::string>& kwargs) const {
  // First check if any of the kwargs are unknown, i.e. don't match the name of
  // any argument in the schema.
  for (const auto& kwarg : kwargs) {
    if (!std::count_if(
            arguments().begin(),
            arguments().end(),
            [&kwarg](const Argument& argument) {
              return argument.name() == kwarg;
            })) {
      return c10::str(
          "Unknown keyword argument '",
          kwarg,
          "' for operator '",
          name(),
          "'. Schema: ",
          *this);
    }
  }
  // If there are unconsumed kwargs but none of them were unknown, the first
  // positional argument present in the kwargs is duplicated.
  for (const auto& argument : arguments()) {
    if (std::find(kwargs.begin(), kwargs.end(), argument.name()) != kwargs.end()) {
      AT_ASSERT(!argument.default_value());
      return c10::str(
          "Argument '",
          argument.name(),
          "' specified both as positional and ",
          "keyword argument. Schema: ",
          *this);
    }
  }
  return "";
}


FunctionSchema FunctionSchema::cloneWithRemappedTypes(
    const std::function<TypePtr(TypePtr)> type_map) const {
  auto update_args = [&](const std::vector<Argument>& args) {
    std::vector<Argument> new_args;
    new_args.reserve(args.size());
    for(const Argument& arg : args) {
      new_args.emplace_back(arg.cloneWithType(type_map(arg.type())));
    }
    return new_args;
  };
  return FunctionSchema(
      name(),
      overload_name(),
      update_args(arguments()),
      update_args(returns()),
      is_vararg(),
      is_varret());
}

// covariant subtyping of list of Arguments
static bool isSubtypeOfList(
    ArrayRef<Argument> child,
    ArrayRef<Argument> parent,
    std::ostream* why_not) {
  if (child.size() != parent.size()) {
    return false;
  }
  for (const auto i : c10::irange(child.size())) {
    const Argument& c = child[i];
    const Argument& p = parent[i];
    if (c.name() != p.name()) {
      return false;
    }
    if (!c.type()->isSubtypeOfExt(*p.type(), why_not)) {
      return false;
    }
  }
  return true;
}

bool FunctionSchema::isSubtypeOf(
    const FunctionSchema& rhs,
    bool as_method,
    std::ostream* why_not) const {
  size_t start = as_method ? 1 : 0;
  // functions are contravariant in arguments but covariant in returns
  return isSubtypeOfList(
             ArrayRef<Argument>(rhs.arguments()).slice(start),
             ArrayRef<Argument>(arguments()).slice(start),
             why_not) &&
      isSubtypeOfList(returns(), rhs.returns(), why_not);
}

} // namespace c10
