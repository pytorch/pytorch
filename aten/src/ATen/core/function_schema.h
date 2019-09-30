#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/operator_name.h>
#include <unordered_map>

namespace c10 {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument;
struct FunctionSchema;

namespace detail {
inline bool defaultValueEquals_(
    const c10::optional<IValue>& lhs,
    const c10::optional<IValue>& rhs) {
  if (lhs.has_value()) {
    return rhs.has_value() && impl::shallowEquals(*lhs, *rhs);
  } else {
    return !rhs.has_value();
  }
}
} // namespace detail

bool operator==(const Argument& lhs, const Argument& rhs);

struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      c10::optional<int32_t> N = c10::nullopt,
      c10::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      c10::optional<AliasInfo> alias_info = c10::nullopt,
      bool is_inferred_type = false)
      : name_(std::move(name)),
        type_(type ? type : TensorType::get()),
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only),
        alias_info_(std::move(alias_info)),
        is_inferred_type_(is_inferred_type) {
    if (default_value_ && default_value_->isTensor()) {
      auto t = default_value_->toTensor();
      AT_ASSERT(!t.defined() || t.is_variable());
    }
  }
  const std::string& name() const {
    return name_;
  }
  TypePtr type() const {
    return type_;
  }
  c10::optional<int32_t> N() const {
    return N_;
  }
  const c10::optional<IValue>& default_value() const {
    return default_value_;
  }
  bool kwarg_only() const {
    return kwarg_only_;
  }
  const c10::optional<AliasInfo>& alias_info() const {
    return alias_info_;
  }
  bool is_inferred_type() const {
    return is_inferred_type_;
  }
  std::string formatTypeMismatchMsg(const std::string& actual_type) const {
    std::string inferred_type_hint;
    if (is_inferred_type()) {
      inferred_type_hint = c10::str(
          "Inferred '",
          name(),
          "' to be of type 'Tensor' ",
          "because it was not annotated with an explicit type.\n");
    }
    return c10::str(
        "Expected a value of type '",
        type()->python_str(),
        "' for argument '",
        name(),
        "' but instead found type '",
        actual_type,
        "'.\n",
        inferred_type_hint);
  }

  Argument cloneWithType(TypePtr new_type) const {
    return Argument(name_, new_type, N_, default_value_, kwarg_only_, alias_info_);
  }

  // this function check whether this Argument is backward compatible with
  // the old one. we consider the following cases are backward compatible:
  //   1) two arguments are equal
  //   2) this arg's type should be subtype of old
  //   3) this arg must provide the same default value if old arg has one,
  bool isBackwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not=nullptr) const;

private:
  std::string name_;
  TypePtr type_;
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  c10::optional<int32_t> N_;

  c10::optional<IValue> default_value_;
  // is this only specifyable as a keyword argument?
  bool kwarg_only_;
  c10::optional<AliasInfo> alias_info_;
  bool is_inferred_type_;
};

inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && *lhs.type() == *rhs.type()
          && lhs.N() == rhs.N()
          && detail::defaultValueEquals_(lhs.default_value(), rhs.default_value())
          && lhs.kwarg_only() == rhs.kwarg_only()
          && lhs.alias_info() == rhs.alias_info();
}

bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs);

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
            std::move(arguments),
            std::move(returns),
            is_vararg,
            is_varret) {}

  // check whether this schema is backward compatible with the old one.
  // the following conditions are considered as this schema is backward
  // compatible with old:
  //   1) two schemas are equal
  //   2) this schema has the same or more positional args than old,
  //      and any positional arg in this schema is backward compatible
  //      with the corresponding one in old schema, which could be an arg
  //      or a kwarg, if it has, or it must provide a default value
  //   3) this schema has the same or more kwargs than old, and all the kwargs
  //      in old schema can find the corresponding kwarg in this schema which
  //      is backward compatible with the old kwarg, and the extra kwargs in
  //      this schema must provide default values.
  bool isBackwardCompatibleWith(
      const FunctionSchema& old,
      std::ostream* why_not=nullptr) const;

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
  bool is_mutable() const {
    return std::any_of(
        arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
          const auto& aliasInfo = arg.alias_info();
          return aliasInfo && aliasInfo.value().isWrite();
        });
  }

  c10::optional<int> argumentIndexWithName(const std::string& name) const {
    for(size_t i = 0; i < arguments().size(); ++i) {
      if(name == arguments()[i].name())
        return i;
    }
    return c10::nullopt;
  }
  FunctionSchema cloneWithArguments(std::vector<Argument> new_arguments) const {
    return FunctionSchema(
        name(),
        overload_name(),
        std::move(new_arguments),
        returns(),
        is_vararg(),
        is_varret());
  }

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

  bool hasAnyAliasInfo() const {
    for (const auto& arg : arguments_) {
      if (arg.alias_info().has_value()) {
        return true;
      }
    }
    for (const auto& ret : returns_) {
      if (ret.alias_info().has_value()) {
        return true;
      }
    }
    return false;
  }

  // can a function with this schema be substituted for a function of rhs's
  // schema and have the program typecheck?
  // as_method - if true, treat this schema as a method and ignore
  // the first argument, which will be the object in both cases
  bool isSubtypeOf(const FunctionSchema& rhs, bool as_method, std::ostream* why_not=nullptr) const;
};

inline bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return lhs.name() == rhs.name()
      && lhs.overload_name() == rhs.overload_name()
      && lhs.arguments() == rhs.arguments()
      && lhs.returns() == rhs.returns()
      && lhs.is_vararg() == rhs.is_vararg()
      && lhs.is_varret() == rhs.is_varret();
}

inline bool operator!=(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return !(lhs == rhs);
}

// print out Argument, which is compatible with FunctionSchema parser
// full format: Type(alias)? name=default_value
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  bool optional_type = arg.type()->kind() == OptionalType::Kind;
  // for adjusting the ? position.
  // in schema, we have Tensor?(a!) input, and t(a!)?.
  // however, t?(a!) doesn't work with schema parser.
  // so we always use Type(alias)? format
  std::stringstream oss;
  if (auto list = arg.type()->cast<c10::ListType>()) {
    oss << list->getElementType()->str();
    oss << "[";
    if (arg.N()) {
      oss << *arg.N();
    }
    oss << "]";
  } else {
    oss << arg.type()->str();
  }
  if (optional_type) {
    oss.seekp(oss.str().size() - 1);
  }
  if (arg.alias_info()) {
    oss << arg.alias_info().value();
  }
  if (optional_type) {
    oss << "?";
  }
  out << oss.str();
  if (!arg.name().empty()) {
    out << " " << arg.name();
  }
  if (arg.default_value()) {
    out << "=";
    if (arg.type()->kind() == c10::TypeKind::StringType) {
        // TODO prettify the result, such as using \n to represent \012
        out << "\'";
        std::ios_base::fmtflags flags(out.flags());
        for (unsigned char c : arg.default_value().value().toStringRef()) {
          out << "\\" << std::oct << std::setfill('0') << std::setw(3)
            << static_cast<uint64_t>(c);
        }
        out.flags(flags);
        out << "\'";
    } else {
      out << arg.default_value().value();
    }
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema);

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

} // namespace c10

#include <ATen/core/function_schema_inl.h>
