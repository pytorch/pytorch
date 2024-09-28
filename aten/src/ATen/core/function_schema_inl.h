#pragma once
#include <ostream>
#include <sstream>

namespace c10 {

template<typename T>
inline void FunctionSchema::checkArg(
    const IValue& value,
    const Argument& argument,
    std::optional<size_t> pos) const {
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
      checkArg<T>(it->second, argument, std::nullopt);
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

} // namespace c10
