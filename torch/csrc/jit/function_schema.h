#pragma once
#include "ATen/ATen.h"

#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/ivalue.h"

namespace torch { namespace jit {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.
struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      at::optional<int32_t> N = at::nullopt,
      at::optional<IValue> default_value = at::nullopt,
      bool kwarg_only = true)
      : name(std::move(name)),
        type(type? type : DynamicType::get()),
        N(N),
        default_value(default_value),
        kwarg_only(kwarg_only) {}
  std::string name;
  TypePtr type;

  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  at::optional<int32_t> N;

  at::optional<IValue> default_value;
  // is this only specifyable as a keyword argument?
  bool kwarg_only;
};

struct FunctionSchema {
  FunctionSchema(
      std::string name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name(std::move(name)),
        arguments(std::move(arguments)),
        returns(std::move(returns)),
        is_vararg(is_vararg),
        is_varret(is_varret) {}
  FunctionSchema(
      Symbol name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : FunctionSchema(
            name.toQualString(),
            std::move(std::move(arguments)),
            std::move(std::move(returns)),
            is_vararg,
            is_varret) {}

  const std::string name;
  const std::vector<Argument> arguments;
  const std::vector<Argument> returns;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primtive' operators whose
  // arguments are not checked by schema
  const bool is_vararg;
  const bool is_varret;
  at::optional<int> argumentIndexWithName(const std::string& name) const {
    for(size_t i = 0; i < arguments.size(); ++i) {
      if(name == arguments[i].name)
        return i;
    }
    return at::nullopt;
  }
};

// for debugging, make sure we can describe the call site
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  return out << arg.type->str() << " " << arg.name << (arg.default_value ? "=<default>" : "");
}

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  out << schema.name;
  out << "(";

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments.size(); ++i) {
    if (i > 0) out << ", ";
    if (schema.arguments[i].kwarg_only && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    out << schema.arguments[i];
  }

  out << ") -> ";
  if (schema.returns.size() == 1) {
    out << schema.returns.at(0).type->str();
  } else if (schema.returns.size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns.size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns[i].type->str();
    }
    out << ")";
  }
  return out;
}

}}
