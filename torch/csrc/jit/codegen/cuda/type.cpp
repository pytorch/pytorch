#include <torch/csrc/jit/codegen/cuda/type.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool isFloatingPointType(DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
      return false;
    case DataType::Double:
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
      return true;
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return false;
    case DataType::Null:
      TORCH_CHECK(
          false, "Null type is not a valid argument to isFloatingPointType");
    default:
      TORCH_CHECK(false, "Type not supported in isFloatingPointType");
  }
}

bool isBooleanType(DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
      return true;
    case DataType::Double:
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
      return false;
    case DataType::Null:
      TORCH_CHECK(false, "Null type is not a valid argument to isBooleanType");
    default:
      TORCH_CHECK(false, "Type not supported in isBooleanType");
  }
}

bool isIntegralType(DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
    case DataType::Double:
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return false;
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
      return true;
    case DataType::Null:
      TORCH_CHECK(
          false, "Null type is not a valid argument to isFloatingPoint");
    default:
      TORCH_CHECK(false, "Type not supported in isFloatingPoint");
  }
}

bool isComplexType(DataType dtype) {
  switch (dtype) {
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return true;
    case DataType::Bool:
    case DataType::Double:
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
    case DataType::Int:
    case DataType::Index:
    case DataType::Int32:
      return false;
    case DataType::Null:
      TORCH_CHECK(false, "Null type is not a valid argument to isComplexType");
    default:
      TORCH_CHECK(false, "Type not supported in isComplexType");
  }
}

bool isIntegerOp(const BinaryOpType bopt) {
  return bopt >= BinaryOpType::Mod && bopt <= BinaryOpType::Rshift;
}

bool isLogicalOp(const BinaryOpType bopt) {
  return bopt >= BinaryOpType::Eq && bopt <= BinaryOpType::NE;
}

bool alsoBooleanOperator(const BinaryOpType bopt) {
  return bopt >= BinaryOpType::And && bopt <= BinaryOpType::Xor;
}

bool alsoBooleanOperator(const UnaryOpType uopt) {
  return uopt >= UnaryOpType::Not && uopt <= UnaryOpType::Not;
}

// Return highest on list (smallest enum val)
DataType promote_type(const DataType& t1, const DataType& t2) {
  TORCH_CHECK(
      DataType::Null != t1 && DataType::Null != t2,
      "Expected promotable DataTypes but got: ",
      t1,
      " and ",
      t2);
  return aten_to_data_type(
      c10::promoteTypes(data_type_to_aten(t1), data_type_to_aten(t2)));
}

// Return highest on list (smallest enum val)
ValType promote_type(const ValType& t1, const ValType& t2) {
  if (t1 == ValType::TensorView || t2 == ValType::TensorView) {
    return ValType::TensorView;
  }
  if (t1 == ValType::Scalar &&
      (t2 == ValType::Scalar || t2 == ValType::NamedScalar)) {
    return ValType::Scalar;
  }
  if (t2 == ValType::Scalar &&
      (t1 == ValType::Scalar || t1 == ValType::NamedScalar)) {
    return ValType::Scalar;
  }
  if (t1 == ValType::NamedScalar && t2 == ValType::NamedScalar) {
    return ValType::Scalar;
  }
  TORCH_CHECK(false, "Expected promotable ValTypes but got: ", t1, " and ", t2);
}

static const char* data_type2string(DataType t) {
  switch (t) {
    case DataType::Bool:
      return "bool";
    case DataType::Double:
      return "double";
    case DataType::Float:
      return "float";
    case DataType::Half:
      return "__half";
    case DataType::BFloat16:
      return "__bfloat";
    case DataType::Int:
      return "int64_t";
    case DataType::Index:
      return "nvfuser_index_t";
    case DataType::Int32:
      return "int";
    case DataType::ComplexFloat:
      return "std::complex<float>";
    case DataType::ComplexDouble:
      return "std::complex<double>";
    case DataType::Null:
      return "null_type";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

static const char* val_type2string(ValType t) {
  switch (t) {
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
    case ValType::Predicate:
      return "Predicate";
    case ValType::TensorIndex:
      return "TensorIndex";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for val type.");
  }
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
    case ExprType::WelfordOp:
      return "WelfordOp";
    case ExprType::MmaOp:
      return "MmaOp";
    case ExprType::TransposeOp:
      return "TransposeOp";
    case ExprType::ShiftOp:
      return "ShiftOp";
    case ExprType::GatherOp:
      return "GatherOp";
    case ExprType::ViewDtypeOp:
      return "ViewDtypeOp";
    case ExprType::ViewOp:
      return "ViewOp";
    case ExprType::Split:
      return "Split";
    case ExprType::Merge:
      return "Merge";
    case ExprType::Allocate:
      return "Allocate";
    case ExprType::BlockSync:
      return "BlockSync";
    case ExprType::GridSync:
      return "GridSync";
    case ExprType::InitMagicZero:
      return "InitMagicZero";
    case ExprType::UpdateMagicZero:
      return "UpdateMagicZero";
    case ExprType::ForLoop:
      return "ForLoop";
    case ExprType::IfThenElse:
      return "IfThenElse";
    case ExprType::GridReduction:
      return "GridReduction";
    case ExprType::GridBroadcast:
      return "GridBroadcast";
    case ExprType::GridWelford:
      return "GridWelford";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for expr type.");
  }
}

bool needFloatSuffix(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Abs:
    case UnaryOpType::Cast:
    case UnaryOpType::Frac:
    case UnaryOpType::Gelu:
    case UnaryOpType::Silu:
    case UnaryOpType::EraseType:
    case UnaryOpType::Neg:
    case UnaryOpType::Relu:
    case UnaryOpType::Reciprocal:
    case UnaryOpType::Set:
    case UnaryOpType::Sigmoid:
      return false;
    default:
      return true;
  }
}

static const char* unary_op_type2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Abs:
      return "abs";
    case UnaryOpType::Acos:
      return "acos";
    case UnaryOpType::Asin:
      return "asin";
    case UnaryOpType::Atan:
      return "atan";
    case UnaryOpType::Atanh:
      return "atanh";
    case UnaryOpType::Cast:
      return "cast";
    case UnaryOpType::Ceil:
      return "ceil";
    case UnaryOpType::Cos:
      return "cos";
    case UnaryOpType::Cosh:
      return "cosh";
    case UnaryOpType::Exp:
      return "exp";
    case UnaryOpType::Expm1:
      return "expm1";
    case UnaryOpType::Erf:
      return "erf";
    case UnaryOpType::Erfc:
      return "erfc";
    case UnaryOpType::Floor:
      return "floor";
    case UnaryOpType::Frac:
      return "frac";
    case UnaryOpType::Silu:
      return "silu";
    case UnaryOpType::Lgamma:
      return "lgamma";
    case UnaryOpType::Log:
      return "log";
    case UnaryOpType::Log10:
      return "log10";
    case UnaryOpType::Log1p:
      return "log1p";
    case UnaryOpType::Log2:
      return "log2";
    case UnaryOpType::EraseType:
      return "erase_type";
    case UnaryOpType::Neg:
      return "neg";
    case UnaryOpType::Not:
      return "not";
    case UnaryOpType::RandLike:
      return "randLike";
    case UnaryOpType::Reciprocal:
      return "reciprocal";
    case UnaryOpType::Relu:
      return "relu";
    case UnaryOpType::Rsqrt:
      return "rsqrt";
    case UnaryOpType::Round:
      return "nearbyint";
    case UnaryOpType::Set:
      return "set";
    case UnaryOpType::Sigmoid:
      return "sigmoid";
    case UnaryOpType::Sin:
      return "sin";
    case UnaryOpType::Sinh:
      return "sinh";
    case UnaryOpType::Sqrt:
      return "sqrt";
    case UnaryOpType::Tan:
      return "tan";
    case UnaryOpType::Tanh:
      return "tanh";
    case UnaryOpType::Trunc:
      return "trunc";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for unary op type.");
  }
}

std::string stringifyBooleanOp(const UnaryOpType uopt) {
  TORCH_INTERNAL_ASSERT(
      uopt == UnaryOpType::Not, uopt, " is not a boolean operator.");
  return "!";
}

static const char* unary_op_type_inline_op2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Neg:
      return "-";
    case UnaryOpType::Not:
      return "~";
    case UnaryOpType::Set:
      return "";
    case UnaryOpType::Address:
      return "(int64_t) &";
    default:
      break;
  }
  return nullptr;
}

bool needFloatSuffix(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Atan2:
    case BinaryOpType::Div:
    case BinaryOpType::Fmod:
      return true;
    default:
      return false;
  }
}

static const char* binary_op_type2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "add";
    case BinaryOpType::Atan2:
      return "atan2";
    case BinaryOpType::Div:
      return "div";
    case BinaryOpType::Fmod:
      return "fmod";
    case BinaryOpType::Max:
      return "fmax";
    case BinaryOpType::Min:
      return "fmin";
    case BinaryOpType::Mul:
      return "mul";
    case BinaryOpType::Pow:
      return "pow";
    case BinaryOpType::Remainder:
      return "remainder";
    case BinaryOpType::Sub:
      return "sub";

    // Integer Ops
    case BinaryOpType::Mod:
      return "mod";
    case BinaryOpType::CeilDiv:
      return "ceilDiv";
    case BinaryOpType::Lshift:
      return "lshift";
    case BinaryOpType::Rshift:
      return "rshift";

    // Logical Ops
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
      TORCH_INTERNAL_ASSERT(false, "No string found for binary op type.");
  }
}

static const char* binary_op_integer_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Max:
      return "max";
    case BinaryOpType::Min:
      return "min";
    case BinaryOpType::Fmod:
      return "fmod";
    default:
      break;
  }
  return nullptr;
}

static const char* binary_op_bool_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Max:
      return "max";
    case BinaryOpType::Min:
      return "min";
    default:
      break;
  }
  return nullptr;
}

static const char* binary_op_type_inline_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "+";
    case BinaryOpType::Div:
      return "/";
    case BinaryOpType::Mul:
      return "*";
    case BinaryOpType::Sub:
      return "-";

    // Integer ops
    case BinaryOpType::Mod:
      return "%";
    case BinaryOpType::Lshift:
      return "<<";
    case BinaryOpType::Rshift:
      return ">>";
    // Logical Ops
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
    // Assume bitwise, otherwise use stringifyBooleanOp
    case BinaryOpType::And:
      return "&";
    case BinaryOpType::Or:
      return "|";
    case BinaryOpType::Xor:
      return "^";
    default:
      break;
  }
  return nullptr;
}

std::string stringifyBooleanOp(const BinaryOpType bopt) {
  switch (bopt) {
    case BinaryOpType::And:
      return "&&";
    case BinaryOpType::Or:
      return "||";
    case BinaryOpType::Xor:
      return "!=";
    default:
      TORCH_INTERNAL_ASSERT(false, bopt, " is not a boolean operator.")
  }
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected TernaryOpType", t);
  }
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
      return "V";
    case ParallelType::MisalignedVectorize:
      return "MV";
    case ParallelType::Unroll:
      return "UR";
    case ParallelType::Unswitch:
      return "US";
    case ParallelType::Mma:
      return "MMA";
    case ParallelType::Serial:
      return "S";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected ParallelType", t);
  }
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected MemoryType", t);
  }
}

static const char* iter_type2string(IterType t) {
  switch (t) {
    case IterType::Iteration:
      return "i";
    case IterType::Reduction:
      return "r";
    case IterType::BroadcastWithStride:
      return "sb";
    case IterType::BroadcastWithoutStride:
      return "b";
    case IterType::Gather:
      return "g";
    case IterType::Stride:
      return "s";
    default:
      // Don't try to print t as it would recursively call this function
      TORCH_INTERNAL_ASSERT(false, "Unexpected IterType");
  }
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected parallel type", t);
  }
}

const unsigned int _WORD_SHIFT = 16;
constexpr unsigned int supported_switch_pair(DataType t1, DataType t2) {
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}

static const char* supported_casts2string(
    const std::pair<DataType, DataType>& t) {
  switch (supported_switch_pair(std::get<0>(t), std::get<1>(t))) {
    case supported_switch_pair(DataType::Index, DataType::Float):
    case supported_switch_pair(DataType::Int, DataType::Float):
    case supported_switch_pair(DataType::Int32, DataType::Float):
    case supported_switch_pair(DataType::Double, DataType::Float):
    case supported_switch_pair(DataType::Bool, DataType::Float):
      return "(float)";
    case supported_switch_pair(DataType::Index, DataType::Int):
    case supported_switch_pair(DataType::Int32, DataType::Int):
    case supported_switch_pair(DataType::Float, DataType::Int):
    case supported_switch_pair(DataType::Double, DataType::Int):
    case supported_switch_pair(DataType::Bool, DataType::Int):
      return "(int64_t)";
    case supported_switch_pair(DataType::Index, DataType::Int32):
    case supported_switch_pair(DataType::Int, DataType::Int32):
    case supported_switch_pair(DataType::Float, DataType::Int32):
    case supported_switch_pair(DataType::Double, DataType::Int32):
    case supported_switch_pair(DataType::Bool, DataType::Int32):
      return "(int32_t)";
    case supported_switch_pair(DataType::Int, DataType::Index):
    case supported_switch_pair(DataType::Int32, DataType::Index):
    case supported_switch_pair(DataType::Float, DataType::Index):
    case supported_switch_pair(DataType::Double, DataType::Index):
      return "(nvfuser_index_t)";
    case supported_switch_pair(DataType::Index, DataType::Double):
    case supported_switch_pair(DataType::Int, DataType::Double):
    case supported_switch_pair(DataType::Int32, DataType::Double):
    case supported_switch_pair(DataType::Float, DataType::Double):
    case supported_switch_pair(DataType::Bool, DataType::Double):
      return "(double)";
    case supported_switch_pair(DataType::Float, DataType::Bool):
    case supported_switch_pair(DataType::Double, DataType::Bool):
    case supported_switch_pair(DataType::Int32, DataType::Bool):
    case supported_switch_pair(DataType::Int, DataType::Bool):
      return "(bool)";
    case supported_switch_pair(DataType::Index, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int32, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Double, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Float, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Bool, DataType::ComplexDouble):
    case supported_switch_pair(DataType::ComplexFloat, DataType::ComplexDouble):
      return "(std::complex<double>)";
    case supported_switch_pair(DataType::Index, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int32, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Double, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Float, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Bool, DataType::ComplexFloat):
    case supported_switch_pair(DataType::ComplexDouble, DataType::ComplexFloat):
      return "(std::complex<float>)";
    case supported_switch_pair(DataType::Float, DataType::Half):
      return "__float2half";
    case supported_switch_pair(DataType::Float, DataType::BFloat16):
      return "__float2bfloat";
    case supported_switch_pair(DataType::Half, DataType::Float):
      return "__half2float";
    case supported_switch_pair(DataType::BFloat16, DataType::Float):
      return "__bfloat2float";
    default:
      return nullptr;
  }
}

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Bool:
      return DataType::Bool;
    case at::ScalarType::Double:
      return DataType::Double;
    case at::ScalarType::Float:
      return DataType::Float;
    case at::ScalarType::Half:
      return DataType::Half;
    case at::ScalarType::BFloat16:
      return DataType::BFloat16;
    case at::ScalarType::Long:
      return DataType::Int;
    case at::ScalarType::Int:
      return DataType::Int32;
    case at::ScalarType::ComplexFloat:
      return DataType::ComplexFloat;
    case at::ScalarType::ComplexDouble:
      return DataType::ComplexDouble;
    default:
      return DataType::Null;
  }
}

at::ScalarType data_type_to_aten(const DataType& data_type) {
  switch (data_type) {
    case DataType::Bool:
      return at::ScalarType::Bool;
    case DataType::Double:
      return at::ScalarType::Double;
    case DataType::Float:
      return at::ScalarType::Float;
    case DataType::Half:
      return at::ScalarType::Half;
    case DataType::BFloat16:
      return at::ScalarType::BFloat16;
    case DataType::Int:
      return at::ScalarType::Long;
    case DataType::Index:
      TORCH_INTERNAL_ASSERT(
          false,
          "Index is determined at compile time,",
          " to convert from an aten type you need to have the compiled information. ",
          "This information is passed to GpuLower at compile time, and then copied to kerned.",
          "There's also this information in FusionExecutorCache and the Registry system.");
    case DataType::Int32:
      return at::ScalarType::Int;
    case DataType::ComplexFloat:
      return at::ScalarType::ComplexFloat;
    case DataType::ComplexDouble:
      return at::ScalarType::ComplexDouble;
    default:
      TORCH_INTERNAL_ASSERT(false, "No data type found for scalar type.");
  }
}

std::ostream& operator<<(std::ostream& out, const ValType vtype) {
  return out << val_type2string(vtype);
}

std::ostream& operator<<(std::ostream& out, const DataType dtype) {
  return out << data_type2string(dtype);
}

std::ostream& operator<<(std::ostream& out, const ExprType etype) {
  return out << expr_type2string(etype);
}

std::ostream& operator<<(std::ostream& out, const UnaryOpType uotype) {
  return out << unary_op_type2string(uotype);
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType botype) {
  return out << binary_op_type2string(botype);
}

std::ostream& operator<<(std::ostream& out, const TernaryOpType totype) {
  return out << ternary_op_type2string(totype);
}

std::ostream& operator<<(std::ostream& out, const ParallelType ptype) {
  return out << stringifyThread(ptype);
}

std::ostream& operator<<(std::ostream& out, const MemoryType mtype) {
  return out << memory_type2string(mtype);
}

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& out,
    const IterType bt) {
  return out << iter_type2string(bt);
}

TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(
    const UnaryOpType uotype) {
  const char* str = unary_op_type_inline_op2string(uotype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

c10::optional<std::string> inline_op_str(const BinaryOpType botype) {
  const char* str = binary_op_type_inline_op2string(botype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

c10::optional<std::string> integer_op_str(const BinaryOpType botype) {
  const char* str = binary_op_integer_op2string(botype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

c10::optional<std::string> bool_op_str(const BinaryOpType botype) {
  const char* str = binary_op_bool_op2string(botype);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

std::string stringifyThreadSize(const ParallelType ptype) {
  return thread_size2string(ptype);
}

std::string stringifyThread(const ParallelType ptype) {
  return parallel_type2string(ptype);
}

std::string typePrefix(const DataType data_type) {
  switch (data_type) {
    case DataType::Bool:
      return "b";
    case DataType::Double:
      return "d";
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
      return "f";
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
      return "i";
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return "c";
    default:
      TORCH_INTERNAL_ASSERT(false, "No data type found for scalar type.");
  }
}

bool isParallelTypeThreadDim(ParallelType ptype) {
  return ptype == ParallelType::TIDx || ptype == ParallelType::TIDy ||
      ptype == ParallelType::TIDz;
}

bool isParallelTypeBlockDim(ParallelType ptype) {
  return ptype == ParallelType::BIDx || ptype == ParallelType::BIDy ||
      ptype == ParallelType::BIDz;
}

bool isParallelTypeThread(ParallelType ptype) {
  return isParallelTypeBlockDim(ptype) || isParallelTypeThreadDim(ptype);
}

bool isParallelTypeVectorize(ParallelType ptype) {
  return ptype == ParallelType::Vectorize ||
      ptype == ParallelType::MisalignedVectorize;
}

c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>& cast) {
  const char* str = supported_casts2string(cast);
  return str != nullptr ? c10::optional<std::string>(std::string(str))
                        : c10::nullopt;
}

size_t dataTypeSize(DataType type) {
  switch (type) {
    case DataType::Bool:
      return sizeof(bool);
    case DataType::ComplexDouble:
      return sizeof(std::complex<double>);
    case DataType::ComplexFloat:
      return sizeof(std::complex<float>);
    case DataType::Double:
      return sizeof(double);
    case DataType::Float:
      return sizeof(float);
    case DataType::Half:
      return sizeof(at::Half);
    case DataType::BFloat16:
      return sizeof(at::BFloat16);
    case DataType::Index:
      TORCH_INTERNAL_ASSERT(
          false, "The actual type of Index is only known at compile time.");
    case DataType::Int:
      return sizeof(uint64_t);
    case DataType::Int32:
      return sizeof(uint32_t);
    default:
      TORCH_INTERNAL_ASSERT(false, "Size undefined for data type, ", type);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
