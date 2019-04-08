#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/alias_info.h>

namespace c10 {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      c10::optional<int32_t> N = c10::nullopt,
      c10::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      c10::optional<AliasInfo> alias_info = c10::nullopt)
      : name_(std::move(name)),
        type_(type ? type : TensorType::get()),
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only),
        alias_info_(std::move(alias_info)) {
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
};

namespace detail {
inline bool defaultValueEquals_(const c10::optional<IValue>& lhs, const c10::optional<IValue>& rhs) {
  if (lhs.has_value()) {
    return rhs.has_value() && shallowEquals(*lhs, *rhs);
  } else {
    return !rhs.has_value();
  }
}
}

inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && lhs.type() == rhs.type()
          && lhs.N() == rhs.N()
          && detail::defaultValueEquals_(lhs.default_value(), rhs.default_value())
          && lhs.kwarg_only() == rhs.kwarg_only()
          && lhs.alias_info() == rhs.alias_info();
}

struct FunctionSchema {
  FunctionSchema(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name_(std::move(name)),
        overload_name_(std::move(overload_name)),
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
      bool is_varret = false,
      std::vector<std::string> writes = {})
      : FunctionSchema(
            name.toQualString(),
            std::move(overload_name),
            std::move(std::move(arguments)),
            std::move(std::move(returns)),
            is_vararg,
            is_varret) {}

private:
  const std::string name_;
  const std::string overload_name_;
  const std::vector<Argument> arguments_;
  const std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primtive' operators whose
  // arguments are not checked by schema
  const bool is_vararg_;
  const bool is_varret_;

public:
  const std::string& name() const {
    return name_;
  }
  const std::string& overload_name() const {
    return overload_name_;
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
    // see [custom operator aliasing]
    const auto kind = Symbol::fromQualString(name_);
    const auto is_custom_op = !kind.is_aten() && !kind.is_prim();
    return is_custom_op ||
        std::any_of(
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
  FunctionSchema withArguments(std::vector<Argument> new_arguments) const {
    return FunctionSchema(
        name(),
        overload_name(),
        std::move(new_arguments),
        returns(),
        is_vararg(),
        is_varret());
  }
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

// for debugging, make sure we can describe the call site
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  return out << arg.type()->str() << " " << arg.name() << (arg.default_value() ? "=<default>" : "");
}

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

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

} // namespace c10
