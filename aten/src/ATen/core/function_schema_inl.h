#pragma once
#include <ostream>
#include <sstream>

// note: windows build doesn't find symbols in operator files unless
// this is a header file

namespace c10 {

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
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

inline size_t findFirstOutArg(const std::vector<Argument>& args) {
  // find the start of out args in the schema
  for (const auto out_start_idx : c10::irange(args.size())) {
    if (args.at(out_start_idx).is_out()) {
      return out_start_idx;
    }
  }
  return args.size();
}

inline bool Argument::isBackwardCompatibleWith(
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

inline bool Argument::isForwardCompatibleWith(
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

inline std::string FunctionSchema::formatTypeMismatchMsg(
    const Argument& expected,
    const std::string& actual_type,
    c10::optional<size_t> position,
    c10::optional<std::string> value) const {
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

inline bool FunctionSchema::isBackwardCompatibleWith(
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

inline bool FunctionSchema::isForwardCompatibleWith(
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
    if (!arguments().at(i).default_value()) {
      if (why_not) {
        why_not
            << "Function schema is not forward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str()
            << " did not provide a default value.";
      }
      return false;
    }

    auto default_val = arguments().at(i).default_value().value();
    if (default_val.isList() || default_val.isGenericDict()) {
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

template<typename T>
inline void FunctionSchema::checkArg(
    const IValue& value,
    const Argument& argument,
    optional<size_t> pos) const {
  if (value.isTensor() && argument.type() == TensorType::get()) {
    // Fast-path for the common case
    return;
  }
  if (!value.type<T>()->isSubtypeOf(*argument.type())) {
    TORCH_CHECK(
        false,
        formatTypeMismatchMsg(
            argument, value.type<T>()->repr_str(), pos));
  }
}

inline std::string FunctionSchema::findErrorInKwargs(const std::vector<std::string>& kwargs) const {
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

template <typename T>
inline void FunctionSchema::checkAndNormalizeInputs(
    std::vector<IValue>& inputs,
    const std::unordered_map<std::string, IValue>& kwargs) const {
  // Do we have more inputs than the schema accepts?
  TORCH_CHECK(
      inputs.size() <= arguments().size(),
      "Expected at most ",
      arguments().size(),
      " argument(s) for operator '",
      name(),
      "', but received ",
      inputs.size(),
      " argument(s). Declaration: ",
      *this);

  size_t consumed_kwargs = 0;
  for (const auto pos : c10::irange(arguments().size())) {
    const auto& argument = arguments()[pos];
    if (pos < inputs.size()) {
      checkArg<T>(inputs[pos], argument, pos);
      continue;
    }
    auto it = kwargs.find(argument.name());
    if (it != kwargs.end()) {
      checkArg<T>(it->second, argument, nullopt);
      inputs.push_back(it->second);
      consumed_kwargs++;
      continue;
    }
    if (argument.default_value()) {
      inputs.push_back(*argument.default_value());
      continue;
    }
    AT_ERROR(
        name(),
        "() is missing value for argument '",
        argument.name(),
        "'. Declaration: ",
        *this);
  }
  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    names.reserve(kwargs.size());
    for(const auto& k : kwargs) {
      names.emplace_back(k.first);
    }
    throw std::runtime_error(findErrorInKwargs(names));
  }
}

inline FunctionSchema FunctionSchema::cloneWithRemappedTypes(
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
inline bool isSubtypeOfList(
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

inline bool FunctionSchema::isSubtypeOf(
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
