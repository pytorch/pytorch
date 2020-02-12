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

static std::unordered_map<DataType, std::string> data_type_string_map {
  {DataType::Float, "Float"},
  {DataType::Int, "Int"}
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

static std::unordered_map<ParallelType, std::string> parallel_type_string_map {
    {ParallelType::BIDz, "BIDz"},
    {ParallelType::BIDy, "BIDy"},
    {ParallelType::BIDx, "BIDx"},
    {ParallelType::TIDz, "TIDz"},
    {ParallelType::TIDy, "TIDy"},
    {ParallelType::TIDx, "TIDx"},
    {ParallelType::Vectorize, "Vectorize"},
    {ParallelType::Unroll, "Unroll"},
    {ParallelType::Serial, "Serial"}
};

static std::unordered_map<at::ScalarType, DataType> at_type_map {
  {at::ScalarType::Float, DataType::Float},
  {at::ScalarType::Int, DataType::Int},
};

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  TORCH_CHECK(at_type_map.count(scalar_type) != 0);
  return at_type_map[scalar_type];
}

std::ostream& operator<<(std::ostream& out, const ValType vtype) {
  TORCH_CHECK(val_type_string_map.count(vtype) != 0);
  return out << val_type_string_map[vtype];
}

std::ostream& operator<<(std::ostream& out, const DataType dtype) {
  TORCH_CHECK(data_type_string_map.count(dtype) != 0);
  return out << data_type_string_map[dtype];
}

std::ostream& operator<<(std::ostream& out, const ExprType etype) {
  TORCH_CHECK(expr_type_string_map.count(etype) != 0);
  return out << expr_type_string_map[etype];
}

std::ostream& operator<<(std::ostream& out, const UnaryOpType uotype) {
  TORCH_CHECK(unary_op_type_string_map.count(uotype) != 0);
  return out << unary_op_type_string_map[uotype];
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType botype) {
  TORCH_CHECK(binary_op_type_string_map.count(botype) != 0);
  return out << binary_op_type_string_map[botype];
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType botype) {
  TORCH_CHECK(binary_op_type_string_map.count(botype) != 0);
  return out << binary_op_type_string_map[botype];
}

std::ostream& operator<<(std::ostream& out, const ParallelType ptype) {
  TORCH_CHECK(parallel_type_string_map.count(ptype) != 0);
  return out << parallel_type_string_map[ptype];
}

}}} // torch::jit::fuser
