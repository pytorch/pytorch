#pragma once

// note: windows build doesn't find symbols in operator files unless
// this is a header file

namespace c10 {

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  out << schema.name();
  if (schema.overload_name() != "") {
    out << "." << schema.overload_name();
  }
  out << "(";

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0) out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    out << schema.arguments()[i];
  }

  if(schema.is_vararg()) {
    if(schema.arguments().size() > 0)
      out << ", ";
    out << "...";
  }

  out << ") -> ";

  const auto& returns = schema.returns();
  out << "(";
  for(size_t i = 0; i < returns.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << returns.at(i);
  }
  if (schema.is_varret()) {
    if (returns.size() != 0) {
      out << ", ";
    }
    out << "...";
  }
  out << ")";
  return out;
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

inline void FunctionSchema::checkArg(
    const IValue& value,
    const Argument& argument,
    optional<size_t> pos) const {
  if (!isSubvalueOf(value, argument.type())) {
    std::string position = pos ? ::c10::str(" in position ", *pos) : "";
    TORCH_CHECK(
        false,
        formatTypeMismatchMsg(
            argument, attemptToRecoverType(value)->python_str(), pos));
  }
}

inline void FunctionSchema::findErrorInKwargs(const std::vector<std::string>& kwargs) const {
  // First check if any of the kwargs are unknown, i.e. don't match the name of
  // any argument in the schema.
  for (const auto& kwarg : kwargs) {
    if (!std::count_if(
            arguments().begin(),
            arguments().end(),
            [&kwarg](const Argument& argument) {
              return argument.name() == kwarg;
            })) {
      throw std::runtime_error(c10::str(
          "Unknown keyword argument '",
          kwarg,
          "' for operator '",
          name(),
          "'. Schema: ",
          *this));
    }
  }
  // If there are unconsumed kwargs but none of them were unknown, the first
  // positional argument present in the kwargs is duplicated.
  for (const auto& argument : arguments()) {
    if (std::find(kwargs.begin(), kwargs.end(), argument.name()) != kwargs.end()) {
      AT_ASSERT(!argument.default_value());
      throw std::runtime_error(c10::str(
          "Argument '",
          argument.name(),
          "' specified both as positional and ",
          "keyword argument. Schema: ",
          *this));
    }
  }
}

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
  for (size_t pos = 0; pos < arguments().size(); ++pos) {
    const auto& argument = arguments()[pos];
    if (pos < inputs.size()) {
      checkArg(inputs[pos], argument, pos);
      continue;
    }
    auto it = kwargs.find(argument.name());
    if (it != kwargs.end()) {
      checkArg(it->second, argument, nullopt);
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
    for(const auto& k : kwargs) {
      names.emplace_back(k.first);
    }
    findErrorInKwargs(names);
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
  for (size_t i = 0; i < child.size(); ++i) {
    const Argument& c = child[i];
    const Argument& p = parent[i];
    if (c.name() != p.name()) {
      return false;
    }
    if (!c.type()->isSubtypeOfExt(p.type(), why_not)) {
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
  // functions are covariant in arguments but contravariant in returns
  return isSubtypeOfList(
             ArrayRef<Argument>(arguments()).slice(start),
             ArrayRef<Argument>(rhs.arguments()).slice(start),
             why_not) &&
      isSubtypeOfList(rhs.returns(), returns(), why_not);
}

} // namespace c10
