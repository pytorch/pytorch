// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// temporary schema description until C10 provides more complete schema information
struct Argument {
  const std::string name;
  // encoded using as_tensor, use tensor_as<T> to get value for attribute
  const at::optional<at::Tensor> default_value;
  // if this can be a graph attribute, the kind of that attribute
  // that matches it
  const at::optional<AttributeKind> attribute_kind;
  // is this a TensorList
  bool is_list;
};

struct OperatorSchema {
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
  if(arg.attribute_kind) {
    switch(*arg.attribute_kind) {
      case AttributeKind::i:
        out << "int64_t ";
        break;
      case AttributeKind::is:
        out << "IntList ";
        break;
      case AttributeKind::f:
        out << "float ";
        break;
      default:
        out << "Tensor ";
    }
  } else if(arg.is_list) {
    out << "TensorList ";
  } else {
    out << "Tensor ";
  }
  out << arg.name;
  if(arg.default_value) {
    out << "=<default>";
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const OperatorSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly of this schema
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

const std::vector<OperatorSchema>& getOperatorSchema(const std::string& name);
std::vector<OperatorSchema> & getOperatorSchemas();

}}
