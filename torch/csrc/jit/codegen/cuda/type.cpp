#include <torch/csrc/jit/codegen/cuda/type.h>

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
  TORCH_CHECK(t1 >= ValType::TensorView && t2 >= ValType::TensorView);
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
  {DataType::Float, "float"},
  {DataType::Int,   "size_t"}
};
static std::unordered_map<ValType, std::string> val_type_string_map {
  {ValType::TensorIndex,  "TensorIndex"},
  {ValType::TensorView,   "TensorView"},
  {ValType::TensorDomain, "TensorDomain"},
  {ValType::IterDomain,   "IterDomain"},
  {ValType::Scalar,       "Scalar"}
};

static std::unordered_map<ExprType, std::string> expr_type_string_map {
    {ExprType::UnaryOp,    "UnaryOp"}
  , {ExprType::BinaryOp,   "BinaryOp"}
  , {ExprType::ForLoop,    "ForLoop"}
  , {ExprType::IfThenElse, "IfThenElse"}
  , {ExprType::Split,      "Split"}
  , {ExprType::Merge,      "Merge"}
  , {ExprType::Reorder,    "Reorder"}
};
static std::unordered_map<UnaryOpType, std::string> unary_op_type_string_map {
    {UnaryOpType::Neg,  "Neg"}
  , {UnaryOpType::Cast, "Cast"}
};
static std::unordered_map<UnaryOpType, std::string> unary_op_type_inline_op_string_map {
    {UnaryOpType::Neg,  "~"}
};
static std::unordered_map<BinaryOpType, std::string> binary_op_type_string_map {
    {BinaryOpType::Add,     "Add"     }
  , {BinaryOpType::Sub,     "Sub"     }
  , {BinaryOpType::Mul,     "Mul"     }
  , {BinaryOpType::Div,     "Div"     }
  , {BinaryOpType::Mod,     "Mod"     }
  , {BinaryOpType::LT,      "LessThan"}
  , {BinaryOpType::CeilDiv, "ceilDiv" }
};
static std::unordered_map<BinaryOpType, std::string> binary_op_type_inline_op_string_map {
    {BinaryOpType::Add,     "+"  }
  , {BinaryOpType::Sub,     "-"  }
  , {BinaryOpType::Mul,     "*"  }
  , {BinaryOpType::Div,     "/"  }
  , {BinaryOpType::Mod,     "%"  }
  , {BinaryOpType::LT,      "<"  }
};

static std::unordered_map<ParallelType, std::string> parallel_type_string_map {
    {ParallelType::BIDz,      "blockIdx.z"},
    {ParallelType::BIDy,      "blockIdx.y"},
    {ParallelType::BIDx,      "blockIdx.x"},
    {ParallelType::TIDz,      "threadIdx.z"},
    {ParallelType::TIDy,      "threadIdx.y"},
    {ParallelType::TIDx,      "threadIdx.x"},
    {ParallelType::Vectorize, "Vectorize"},
    {ParallelType::Unroll,    "Unroll"},
    {ParallelType::Serial,    "Serial"}
};

static std::unordered_map<at::ScalarType, DataType> at_type_map {
  {at::ScalarType::Float, DataType::Float},
  {at::ScalarType::Int, DataType::Int},
};

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  TORCH_INTERNAL_ASSERT(at_type_map.count(scalar_type) != 0,
  "No string found for scalar type.");
  return at_type_map[scalar_type];
}

std::ostream& operator<<(std::ostream& out, const ValType vtype) {
  TORCH_INTERNAL_ASSERT(val_type_string_map.count(vtype) != 0,
  "No string found for val type.");
  return out << val_type_string_map[vtype];
}

std::ostream& operator<<(std::ostream& out, const DataType dtype) {
  TORCH_INTERNAL_ASSERT(data_type_string_map.count(dtype) != 0,
  "No string found for data type.");
  return out << data_type_string_map[dtype];
}

std::ostream& operator<<(std::ostream& out, const ExprType etype) {
  TORCH_INTERNAL_ASSERT(expr_type_string_map.count(etype) != 0,
  "No string found for expr type.");
  return out << expr_type_string_map[etype];
}

std::ostream& operator<<(std::ostream& out, const UnaryOpType uotype) {
  TORCH_INTERNAL_ASSERT(unary_op_type_string_map.count(uotype) != 0,
  "No string found for UnaryOp type.");
  return out << unary_op_type_string_map[uotype];
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType botype) {
  TORCH_INTERNAL_ASSERT(binary_op_type_string_map.count(botype) != 0,
  "No string found for BinaryOp type.");
  return out << binary_op_type_string_map[botype];
}

std::ostream& operator<<(std::ostream& out, const ParallelType ptype) {
  TORCH_INTERNAL_ASSERT(parallel_type_string_map.count(ptype) != 0,
  "No string found for parallel type.");
  return out << parallel_type_string_map[ptype];
}

c10::optional<std::string> inline_op_str(const UnaryOpType uotype) {
  if(unary_op_type_inline_op_string_map.find(uotype) == unary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return unary_op_type_inline_op_string_map[uotype];
  }
}

c10::optional<std::string> inline_op_str(const BinaryOpType botype) {
  if(binary_op_type_inline_op_string_map.find(botype) == binary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return binary_op_type_inline_op_string_map[botype];
  }
}

}}} // torch::jit::fuser
