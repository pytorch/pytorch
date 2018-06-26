#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are availiable
struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      at::optional<int32_t> N = at::nullopt,
      at::optional<at::Tensor> default_value = at::nullopt,
      bool kwarg_only = true)
      : name(std::move(name)),
        type(type),
        N(N),
        default_value(default_value),
        kwarg_only(kwarg_only) {}
  std::string name;
  TypePtr type;
  at::optional<int32_t> N; // a fixed length for list types, at::nullopt means the list is dynamically sized
  // encoded using as_tensor, use tensor_as<T> to get value for attribute
  at::optional<at::Tensor> default_value;
  // is this only specifyable as a keyword argument?
  bool kwarg_only;
};

struct FunctionSchema {
  const std::string name;
  const std::vector<Argument> arguments;
  const std::vector<Argument> returns;

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
  auto emitList = [&](const std::vector<Argument>& args) {
    out << "(";
    for(size_t i = 0; i < args.size(); ++i) {
      if(i > 0)
        out << ", ";
      out << args[i];
    }
    out << ")";
  };

  out << schema.name;
  emitList(schema.arguments);
  if(schema.returns.size() > 1) {
    out << " -> ";
    emitList(schema.returns);
  }
  return out;
}

const std::vector<FunctionSchema>& getFunctionSchema(const std::string& name);
std::vector<FunctionSchema> & getFunctionSchemas();

}}
