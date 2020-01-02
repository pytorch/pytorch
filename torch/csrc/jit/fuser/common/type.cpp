#include <torch/csrc/jit/fuser/common/type.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

static std::unordered_map<ValType, std::string> val_type_string_map {
  {ValType::Float, "Float"}
};
static std::unordered_map<ExprType, std::string> expr_type_string_map {
  {ExprType::Add, "Add"}
};

std::string stringify(const ValType valtype) {
  return val_type_string_map[valtype];
}

std::string stringify(const ExprType exprtype) {
  return expr_type_string_map[exprtype];
}

std::ostream& operator<<(std::ostream& out, const ValType valtype) {
  return out << stringify(valtype);
}

std::ostream& operator<<(std::ostream& out, const ExprType exprtype) {
  return out << stringify(exprtype);
}

}}} // torch::jit::fuser
