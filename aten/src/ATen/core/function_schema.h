#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/operator_name.h>
#include <c10/util/Optional.h>
#include <unordered_map>
#include <memory>

namespace c10 {
class IValue;
struct Type;
using TypePtr = std::shared_ptr<Type>;

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument;

struct FunctionSchema {
  FunctionSchema(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name_({std::move(name), std::move(overload_name)}),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {}

  FunctionSchema(
      Symbol name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : FunctionSchema(
            name.toQualString(),
            std::move(overload_name),
            std::move(std::move(arguments)),
            std::move(std::move(returns)),
            is_vararg,
            is_varret) {}

private:
  OperatorName name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primtive' operators whose
  // arguments are not checked by schema
  bool is_vararg_;
  bool is_varret_;
  void checkArg(const IValue& value, const Argument& argument, optional<size_t> pos) const;

public:
  const OperatorName& operator_name() const {
    return name_;
  }
  const std::string& name() const {
    return name_.name;
  }
  const std::string& overload_name() const {
    return name_.overload_name;
  }
  const std::vector<Argument>& arguments() const {
    return arguments_;
  }
  const std::vector<Argument>& returns() const {
    return returns_;
  }
  bool is_vararg() const {
    return is_vararg_;
  }
  bool is_varret() const {
    return is_varret_;
  }
  bool is_mutable() const;

  c10::optional<int> argumentIndexWithName(const std::string& name) const;
  FunctionSchema cloneWithArguments(std::vector<Argument> new_arguments) const;

  std::string formatTypeMismatchMsg(
      const Argument& expected,
      const std::string& actual_type,
      c10::optional<size_t> position = c10::nullopt,
      c10::optional<std::string> value = c10::nullopt) const;

  FunctionSchema cloneWithRemappedTypes(
      const std::function<TypePtr(TypePtr)> type_map) const;

  // Check that inputs have the correct types and appends any missing default
  // values.
  void checkAndNormalizeInputs(
      std::vector<IValue>& inputs,
      const std::unordered_map<std::string, IValue>& kwargs) const;

  void findErrorInKwargs(const std::vector<std::string>& kwargs) const;

  bool hasAnyAliasInfo() const;
};

bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs);
bool operator!=(const FunctionSchema& lhs, const FunctionSchema& rhs);

// print out Argument, which is compatible with FunctionSchema parser
// full format: Type(alias)? name=default_value
std::ostream& operator<<(std::ostream& out, const Argument& arg);

std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema);

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

} // namespace c10

#include <ATen/core/function_schema_inl.h>
