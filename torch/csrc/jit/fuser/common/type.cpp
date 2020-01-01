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

// bool is_scalar(const CType& type){
//   if(type<CType::kStatement)
//     return true;
//   return false;
// }

// CType promote(const CType& t1, const CType& t2){
//   assert(
//     (t1 < CType::kStatement && t2 < CType::kStatement) ||
//     (t1 > CType::kStatement && t2 > CType::kStatement)
//   );
//   return(t1 < t2 ? t1 : t2);
// }

// bool is_scalar(const DType& type){
//   return is_scalar(type.ctype());
// }

// DType promote(const DType& t1, const DType& t2){
//   assert(t1.lanes() == t2.lanes());
//   return DType(promote(t1.ctype(), t2.ctype()), t1.lanes());
// }


}}} // torch::jit::fuser
