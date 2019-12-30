#include <torch/csrc/jit/fuser/common/ir.h>

#include <iostream>
#include <unordered_map>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

static std::unordered_map<ValType, std::string> val_type_print_map {
  {ValType::Expr, "Expr"}
, {ValType::Scalar, "Scalar"}
};

std::ostream& operator<<(std::ostream& out, const ValType valtype) {
  const auto iter = val_type_print_map.find(valtype);
  if (iter == val_type_print_map.end()) {
    out << "unknown val type";
  } else {
    out << val_type_print_map[valtype];
  }

  return out;
}

template <typename T>
int Val::dispatch(T& handler) {
  if (type_ == ValType::Expr) {
    return handler.handle(this, static_cast<Expr*>(contained_));
  }

  return handler.handle(this, contained_);
}

// TODO: document this requirement
template int Val::dispatch(SampleValHandler&);

template <typename T>
int SampleValHandler::handle(Val* val, T* contained) {
  return -1;
}

// TODO: create macro that instantiates templates for all types
template int SampleValHandler::handle(Val* val, void* contained);

int SampleValHandler::handle(Val* val, Expr* expr) {
  return 0;
}


}}} // torch::jit::fuser
