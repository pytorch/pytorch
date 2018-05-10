#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are availiable
struct Argument {
  const std::string name;
  const TypePtr type;
  // encoded using as_tensor, use tensor_as<T> to get value for attribute
  const at::optional<at::Tensor> default_value;
  // if this can be a graph attribute, the kind of that attribute
  // that matches it
  const at::optional<AttributeKind> attribute_kind;
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
  // if can report more friendly types if we have an attribute
  if(arg.attribute_kind) {
    switch(*arg.attribute_kind) {
      case AttributeKind::i:
        out << "int64_t";
        break;
      case AttributeKind::is:
        out << "IntList";
        break;
      case AttributeKind::f:
        out << "float";
        break;
      default:
        out << arg.type->name();
        break;
    }
  } else {
    out << arg.type->name();
  }
  out << " " << arg.name;
  if(arg.default_value) {
    out << "=<default>";
  }
  return out;
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
