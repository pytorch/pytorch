#pragma once

// note: windows build doesn't find symbols in operator files unless
// this is a header file

namespace c10 {

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  out << schema.name();
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
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out;
}

inline void FunctionSchema::checkArg(const IValue& value, const Argument& argument, optional<size_t> pos) const {
  if (!isSubvalueOf(value, argument.type())) {
    std::string position = pos ? ::c10::str(" in position ", *pos) : "";
    AT_ERROR(
        "Expected value of type ",
        *argument.type(),
        " for argument '",
        argument.name(),
        "'",
        position,
        ", but instead got value of type ",
        attemptToRecoverType(value)->str(),
        ". Declaration: ",
        *this);
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
  AT_CHECK(
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

} // namespace c10
