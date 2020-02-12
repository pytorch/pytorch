#include <torch/csrc/jit/fuser/common/type.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

//Return highest on list (smallest enum val)
DataType promote_type(const DataType& t1, const DataType& t2){
  TORCH_CHECK(DataType::Null != t1 && DataType::Null != t2);
  return t1 < t2 ? t1 : t2;
}

//Return highest on list (smallest enum val)
ValType promote_type(const ValType& t1, const ValType& t2){
  //Check that it's a promotable type (with dtype)
  //static_assert??
  return t1 < t2 ? t1 : t2;
}

bool is_cast_legal(const DataType& t1, const DataType& t2) {
  if((DataType::Null == t1) || (DataType::Null == t2)) return false;
  // In theory there could be stronger real check here in the future
  return true;
}

static std::unordered_map<DataType, std::string> data_type_string_map {
  {DataType::Float, "Float"},
  {DataType::Int,   "Int"}
};
static std::unordered_map<ValType, std::string> val_type_string_map {
  {ValType::Tensor, "Tensor"},
  {ValType::TensorView, "TensorView"},
  {ValType::TensorDomain, "TensorDomain"},
  {ValType::IterDomain, "IterDomain"},
  {ValType::Scalar, "Scalar"}
};

static std::unordered_map<ExprType, std::string> expr_type_string_map {
    {ExprType::UnaryOp,  "UnaryOp"}
  , {ExprType::BinaryOp, "BinaryOp"}
};
static std::unordered_map<UnaryOpType, std::string> unary_op_type_string_map {
    {UnaryOpType::Neg,  "Neg"}
  , {UnaryOpType::Cast, "Cast"}
};
static std::unordered_map<BinaryOpType, std::string> binary_op_type_string_map {
    {BinaryOpType::Add, "Add"}
  , {BinaryOpType::Sub, "Sub"}
  , {BinaryOpType::Mul, "Mul"}
  , {BinaryOpType::Div, "Div"}
  , {BinaryOpType::Mod, "Mod"}
};

std::string stringify(const DataType datatype) {
  return data_type_string_map[datatype];
}

std::string stringify(const ValType valtype) {
  return val_type_string_map[valtype];
}

std::string stringify(const ExprType exprtype) {
  return expr_type_string_map[exprtype];
}

std::string stringify(const UnaryOpType type) {
  return unary_op_type_string_map[type];
}

std::string stringify(const BinaryOpType type) {
  return binary_op_type_string_map[type];
}

std::ostream& operator<<(std::ostream& out, const DataType datatype) {
  return out << stringify(datatype);
}

std::ostream& operator<<(std::ostream& out, const ValType valtype) {
  return out << stringify(valtype);
}

std::ostream& operator<<(std::ostream& out, const ExprType exprtype) {
  return out << stringify(exprtype);
}

std::ostream& operator<<(std::ostream& out, const UnaryOpType type) {
  return out << stringify(type);
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType type) {
  return out << stringify(type);
}

}}} // torch::jit::fuser
