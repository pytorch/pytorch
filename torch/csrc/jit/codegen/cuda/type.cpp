#include <torch/csrc/jit/codegen/cuda/type.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// Return highest on list (smallest enum val)
DataType promote_type(const DataType& t1, const DataType& t2) {
  TORCH_CHECK(
      DataType::Null != t1 && DataType::Null != t2,
      "Expected promotable DataTypes but got: ",
      t1,
      " and ",
      t2);
  return t1 < t2 ? t1 : t2;
}

// Return highest on list (smallest enum val)
ValType promote_type(const ValType& t1, const ValType& t2) {
  TORCH_CHECK(
      t1 >= ValType::TensorView && t2 >= ValType::TensorView,
      "Expected promotable ValTypes but got: ",
      t1,
      " and ",
      t2);
  // Check that it's a promotable type (with dtype)
  // static_assert??
  return t1 < t2 ? t1 : t2;
}

static const char* data_type2string(DataType t) {
  switch (t) {
    case DataType::Bool:
      return "bool";
    case DataType::Float:
      return "float";
    case DataType::Half:
      return "__half";
    case DataType::Int:
      return "size_t";
    case DataType::Null:
      return "nullptr";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

static const char* val_type2string(ValType t) {
  switch (t) {
    case ValType::TensorIndex:
      return "TensorIndex";
    case ValType::TensorView:
      return "TensorView";
    case ValType::TensorDomain:
      return "TensorDomain";
    case ValType::IterDomain:
      return "IterDomain";
    case ValType::Scalar:
      return "Scalar";
    case ValType::NamedScalar:
      return "NamedScalar";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for val type.");
  return nullptr;
}

static const char* expr_type2string(ExprType t) {
  switch (t) {
    case ExprType::UnaryOp:
      return "UnaryOp";
    case ExprType::BinaryOp:
      return "BinaryOp";
    case ExprType::TernaryOp:
      return "TernaryOp";
    case ExprType::ReductionOp:
      return "ReductionOp";
    case ExprType::BroadcastOp:
      return "BroadcastOp";
    case ExprType::ForLoop:
      return "ForLoop";
    case ExprType::IfThenElse:
      return "IfThenElse";
    case ExprType::Allocate:
      return "Allocate";
    case ExprType::Split:
      return "Split";
    case ExprType::Merge:
      return "Merge";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for expr type.");
  return nullptr;
}

static const char* unary_op_type2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Abs:
      return "fabs";
    case UnaryOpType::Acos:
      return "acosf";
    case UnaryOpType::Asin:
      return "asinf";
    case UnaryOpType::Atan:
      return "atanf";
    case UnaryOpType::Atanh:
      return "atanhf";
    case UnaryOpType::Cast:
      return "cast";
    case UnaryOpType::Ceil:
      return "ceilf";
    case UnaryOpType::Cos:
      return "cosf";
    case UnaryOpType::Cosh:
      return "coshf";
    case UnaryOpType::Exp:
      return "expf";
    case UnaryOpType::Expm1:
      return "expm1f";
    case UnaryOpType::Erf:
      return "erff";
    case UnaryOpType::Erfc:
      return "erfcf";
    case UnaryOpType::Floor:
      return "floorf";
    case UnaryOpType::Frac:
      return "frac";
    case UnaryOpType::Gelu:
      return "gelu";
    case UnaryOpType::Lgamma:
      return "lgammaf";
    case UnaryOpType::Log:
      return "logf";
    case UnaryOpType::Log10:
      return "log10f";
    case UnaryOpType::Log1p:
      return "log1pf";
    case UnaryOpType::Log2:
      return "log2f";
    case UnaryOpType::Neg:
      return "neg";
    case UnaryOpType::RandLike:
      return "randLike";
    case UnaryOpType::Reciprocal:
      return "reciprocal";
    case UnaryOpType::Relu:
      return "relu";
    case UnaryOpType::Rsqrt:
      return "rsqrtf";
    case UnaryOpType::Round:
      return "roundf";
    case UnaryOpType::Set:
      return "set";
    case UnaryOpType::Sigmoid:
      return "sigmoid";
    case UnaryOpType::Sin:
      return "sinf";
    case UnaryOpType::Sinh:
      return "sinhf";
    case UnaryOpType::Sqrt:
      return "sqrtf";
    case UnaryOpType::Tan:
      return "tanf";
    case UnaryOpType::Tanh:
      return "tanhf";
    case UnaryOpType::Trunc:
      return "truncf";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for unary op type.");
  return nullptr;
}

static const char* unary_op_type_inline_op2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Neg:
      return "-";
    case UnaryOpType::Set:
      return "";
    default:
      break;
  }
  return nullptr;
}

static const char* binary_op_type2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "add";
    case BinaryOpType::Atan2:
      return "atan2f";
    case BinaryOpType::Div:
      return "div";
    case BinaryOpType::Fmod:
      return "fmodf";
    case BinaryOpType::Max:
      return "fmaxf";
    case BinaryOpType::Min:
      return "fminf";
    case BinaryOpType::Mul:
      return "mul";
    case BinaryOpType::Pow:
      return "powf";
    case BinaryOpType::Remainder:
      return "remainder";
    case BinaryOpType::Sub:
      return "sub";

    // Logical Ops
    case BinaryOpType::Mod:
      return "mod";
    case BinaryOpType::CeilDiv:
      return "ceilDiv";
    case BinaryOpType::And:
      return "and";
    case BinaryOpType::Eq:
      return "equal";
    case BinaryOpType::GE:
      return "greaterThanOrEqual";
    case BinaryOpType::GT:
      return "greaterThan";
    case BinaryOpType::LE:
      return "lessThanOrEqual";
    case BinaryOpType::LT:
      return "lessThan";
    case BinaryOpType::NE:
      return "notEqual";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for binary op type.");
  return nullptr;
}

static const char* binary_op_type_inline_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "+";
    case BinaryOpType::Div:
      return "/";
    case BinaryOpType::Mod:
      return "%";
    case BinaryOpType::Mul:
      return "*";
    case BinaryOpType::Sub:
      return "-";

    // Logical Ops
    case BinaryOpType::And:
      return "&&";
    case BinaryOpType::Eq:
      return "==";
    case BinaryOpType::GE:
      return ">=";
    case BinaryOpType::GT:
      return ">";
    case BinaryOpType::LE:
      return "<=";
    case BinaryOpType::LT:
      return "<";
    case BinaryOpType::NE:
      return "!=";
    default:
      break;
  }
  return nullptr;
}

static const char* ternary_op_type2string(TernaryOpType t) {
  switch (t) {
    case TernaryOpType::Clamp:
      return "clamp";
    case TernaryOpType::Threshold:
      return "threshold";
    case TernaryOpType::Where:
      return "where";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for ternary op type.");
  return nullptr;
}

static const char* parallel_type2string(ParallelType t) {
  switch (t) {
    case ParallelType::BIDz:
      return "blockIdx.z";
    case ParallelType::BIDy:
      return "blockIdx.y";
    case ParallelType::BIDx:
      return "blockIdx.x";
    case ParallelType::TIDz:
      return "threadIdx.z";
    case ParallelType::TIDy:
      return "threadIdx.y";
    case ParallelType::TIDx:
      return "threadIdx.x";
    case ParallelType::Vectorize:
      return "Vectorize";
    case ParallelType::Unroll:
      return "Unroll";
    case ParallelType::Serial:
      return "Serial";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for parallel type.");
  return nullptr;
}

static const char* memory_type2string(MemoryType t) {
  switch (t) {
    case MemoryType::Local:
      return "register";
    case MemoryType::Shared:
      return "shared";
    case MemoryType::Global:
      return "global";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for memory type.");
  return nullptr;
}

static DataType at_type2data_type(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Bool:
      return DataType::Bool;
    case at::ScalarType::Float:
      return DataType::Float;
    case at::ScalarType::Half:
      return DataType::Half;
    case at::ScalarType::Int:
      return DataType::Int;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No data type found for scalar type.");
  return DataType::Null;
}

static const char* thread_size2string(ParallelType t) {
  switch (t) {
    case ParallelType::BIDz:
      return "gridDim.z";
    case ParallelType::BIDy:
      return "gridDim.y";
    case ParallelType::BIDx:
      return "gridDim.x";
    case ParallelType::TIDz:
      return "blockDim.z";
    case ParallelType::TIDy:
      return "blockDim.y";
    case ParallelType::TIDx:
      return "blockDim.x";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Could not find size of the thread type ", t);
  return nullptr;
}

const unsigned int _WORD_SHIFT = 16;
constexpr unsigned int supported_switch_pair(DataType t1, DataType t2) {
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}
static const char* supported_casts2string(
    const std::pair<DataType, DataType>& t) {
  switch (supported_switch_pair(std::get<0>(t), std::get<1>(t))) {
    case supported_switch_pair(DataType::Float, DataType::Half):
      return "__float2half";
    case supported_switch_pair(DataType::Half, DataType::Float):
      return "__half2float";
    default:
      break;
  }
  return nullptr;
}

bool is_logical_op(const BinaryOpType& bot) {
  switch (bot) {
    case BinaryOpType::And:
    case BinaryOpType::Eq:
    case BinaryOpType::GE:
    case BinaryOpType::GT:
    case BinaryOpType::LE:
    case BinaryOpType::LT:
    case BinaryOpType::NE:
      return true;
    default:
      return false;
  }
}

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  return at_type2data_type(scalar_type);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ValType vtype) {
  return out << val_type2string(vtype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const DataType dtype) {
  return out << data_type2string(dtype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ExprType etype) {
  return out << expr_type2string(etype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const UnaryOpType uotype) {
  return out << unary_op_type2string(uotype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const BinaryOpType botype) {
  return out << binary_op_type2string(botype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const TernaryOpType totype) {
  return out << ternary_op_type2string(totype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ParallelType ptype) {
  return out << parallel_type2string(ptype);
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const MemoryType mtype) {
  return out << memory_type2string(mtype);
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const UnaryOpType uotype) {
  const char* str = unary_op_type_inline_op2string(uotype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const BinaryOpType botype) {
  const char* str = binary_op_type_inline_op2string(botype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

std::string stringifyThreadSize(const ParallelType ptype) {
  return thread_size2string(ptype);
}

TORCH_CUDA_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>& cast) {
  const char* str = supported_casts2string(cast);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

} // namespace fuser
} // namespace jit
} // namespace torch
