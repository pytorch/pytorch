#include <torch/csrc/jit/codegen/cuda/type.h>

#include <ATen/cuda/CUDAContext.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

DataType indexModeToDtype(KernelIndexMode index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return DataType::Int32;
    case KernelIndexMode::INT64:
      return DataType::Int;
    default:
      TORCH_CHECK(false, "Invalid kernel index mode type.");
  }
}

bool isFloatingPointType(DataType dtype) {
  switch (dtype) {
    case DataType::Double:
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
      return true;
    case DataType::Bool:
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
      TORCH_CHECK(false, "Null type is not a valid argument to isIntegralType");
    default:
      TORCH_CHECK(false, "Type not supported in isIntegralType");
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

bool isVectorType(DataType dtype) {
  switch (dtype) {
    case DataType::Float_2:
    case DataType::Double_2:
      return true;
    default:
      return false;
  }
}

DataType getVectorType(DataType dtype, size_t vec_size) {
  switch (dtype) {
    case DataType::Float:
      TORCH_INTERNAL_ASSERT(vec_size == 2, "Not supported vectorized type");
      return DataType::Float_2;
    case DataType::Double:
      TORCH_INTERNAL_ASSERT(vec_size == 2, "Not supported vectorized type");
      return DataType::Double_2;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Not supported vectorized type:", dtype, " and ", vec_size);
  }
}

int getVectorSizeFromType(DataType dtype) {
  switch (dtype) {
    case DataType::Float_2:
    case DataType::Double_2:
      return 2;
    default:
      TORCH_INTERNAL_ASSERT(false, "Not a vector type:", dtype);
  }
}

DataType getTypeFromVectorType(DataType dtype) {
  switch (dtype) {
    case DataType::Float_2:
      return DataType::Float;
    case DataType::Double_2:
      return DataType::Double;
    default:
      TORCH_INTERNAL_ASSERT(false, "Not a vector type:", dtype);
  }
}

DataType getTypeFromComplexType(DataType dtype) {
  switch (dtype) {
    case DataType::ComplexFloat:
      return DataType::Float;
    case DataType::ComplexDouble:
      return DataType::Double;
    default:
      TORCH_INTERNAL_ASSERT(false, "Not a vector type:", dtype);
  }
}

bool isSupportedTypeByDevice(DataType dtype) {
  auto prop = at::cuda::getCurrentDeviceProperties();
  auto major_ver = prop->major;
  switch (dtype) {
    case DataType::BFloat16:
      return major_ver >= 8;
    default:
      return true;
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
    case DataType::Double_2:
      return "Array<double, 2, 1>";
    case DataType::Float_2:
      return "Array<float, 2, 1>";
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
    case ValType::IntPair:
      return "IntPair";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for val type.");
  }
}

static const char* predicate_type2string(PredicateType t) {
  switch (t) {
    case PredicateType::Manual:
      return "Manual";
    case PredicateType::Inline:
      return "Inline";
    case PredicateType::Unswitch:
      return "Unswitch";
    case PredicateType::Vectorize:
      return "Vectorize";
    case PredicateType::Misaligned:
      return "Misaligned";
    case PredicateType::Shift:
      return "Shift";
    case PredicateType::Padding:
      return "Padding";
    case PredicateType::ReductionWrite:
      return "ReductionWrite";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for predicate type.");
  }
}

static const char* expr_type2string(ExprType t) {
  switch (t) {
    case ExprType::FullOp:
      return "FullOp";
    case ExprType::ARangeOp:
      return "ARangeOp";
    case ExprType::EyeOp:
      return "EyeOp";
    case ExprType::UnaryOp:
      return "UnaryOp";
    case ExprType::BinaryOp:
      return "BinaryOp";
    case ExprType::TernaryOp:
      return "TernaryOp";
    case ExprType::RNGOp:
      return "RNGOp";
    case ExprType::ReductionOp:
      return "ReductionOp";
    case ExprType::GroupedReductionOp:
      return "GroupedReductionOp";
    case ExprType::BroadcastOp:
      return "BroadcastOp";
    case ExprType::WelfordOp:
      return "WelfordOp";
    case ExprType::GroupedWelfordOp:
      return "GroupedWelfordOp";
    case ExprType::LoadStoreOp:
      return "LoadStoreOp";
    case ExprType::MmaOp:
      return "MmaOp";
    case ExprType::TransposeOp:
      return "TransposeOp";
    case ExprType::ExpandOp:
      return "ExpandOp";
    case ExprType::ShiftOp:
      return "ShiftOp";
    case ExprType::GatherOp:
      return "GatherOp";
    case ExprType::ViewAsScalar:
      return "ViewAsScalar";
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
    case ExprType::CpAsyncWait:
      return "CpAsyncWait";
    case ExprType::CpAsyncCommit:
      return "CpAsyncCommit";
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
    case ExprType::GroupedGridReduction:
      return "GroupedGridReduction";
    case ExprType::GridBroadcast:
      return "GridBroadcast";
    case ExprType::GridWelford:
      return "GridWelford";
    case ExprType::GroupedGridWelford:
      return "GroupedGridWelford";
    case ExprType::Swizzle2D:
      return "Swizzle2D";
    case ExprType::Swizzle2DInt:
      return "Swizzle2DInt";
    case ExprType::PairSelect:
      return "PairSelect";
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
    case UnaryOpType::BitCast:
    case UnaryOpType::Neg:
    case UnaryOpType::Relu:
    case UnaryOpType::Reciprocal:
    case UnaryOpType::Set:
    case UnaryOpType::Sigmoid:
    case UnaryOpType::IsFinite:
    case UnaryOpType::IsInf:
    case UnaryOpType::IsNan:
    case UnaryOpType::IsNegInf:
    case UnaryOpType::IsPosInf:
    case UnaryOpType::IsReal:
    case UnaryOpType::Print:
      return false;
    default:
      return true;
  }
}

bool needFloatSuffix(RNGOpType t) {
  return true;
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
    case UnaryOpType::BitCast:
      return "erase_type";
    case UnaryOpType::Neg:
      return "neg";
    case UnaryOpType::Not:
      return "not";
    case UnaryOpType::Print:
      return "print";
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
    case UnaryOpType::IsFinite:
      return "isfinite";
    case UnaryOpType::IsInf:
      return "isinf";
    case UnaryOpType::IsNan:
      return "isnan";
    case UnaryOpType::IsNegInf:
      return "isneginf";
    case UnaryOpType::IsPosInf:
      return "isposinf";
    case UnaryOpType::IsReal:
      return "isreal";
    case UnaryOpType::Real:
      return "std::real";
    case UnaryOpType::Imag:
      return "std::imag";
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

static const char* rng_op_type_inline_op2string(RNGOpType t) {
  switch (t) {
    case RNGOpType::Uniform:
      return "rng_uniform";
    case RNGOpType::UniformRange:
      return "rng_uniform_range";
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
    case TernaryOpType::Lerp:
      return "lerp";
    case TernaryOpType::Threshold:
      return "threshold";
    case TernaryOpType::Where:
      return "where";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected TernaryOpType");
  }
}

static const char* rng_op_type2string(RNGOpType t) {
  switch (t) {
    case RNGOpType::Uniform:
      return "rng_uniform";
    case RNGOpType::UniformRange:
      return "rng_uniform_range";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected RNGOpType");
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
    case ParallelType::Group:
      return "G";
    case ParallelType::Serial:
      return "S";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected ParallelType");
  }
}

std::unordered_set<ParallelType> allParallelTypesExcept(
    const std::unordered_set<ParallelType>& except) {
  std::unordered_set<ParallelType> result = {
      ParallelType::BIDz,
      ParallelType::BIDy,
      ParallelType::BIDx,
      ParallelType::TIDz,
      ParallelType::TIDy,
      ParallelType::TIDx,
      ParallelType::Vectorize,
      ParallelType::MisalignedVectorize,
      ParallelType::Unroll,
      ParallelType::Unswitch,
      ParallelType::Mma,
      ParallelType::Group,
      ParallelType::Serial};
  for (auto t : except) {
    result.erase(t);
  }
  return result;
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected MemoryType");
  }
}

static const char* id_map_mode_type2string(IdMappingMode t) {
  switch (t) {
    case IdMappingMode::PERMISSIVE:
      return "permissive";
    case IdMappingMode::EXACT:
      return "exact";
    case IdMappingMode::LOOP:
      return "loop";
    default:
      // Don't try to print t as it would recursively call this function
      TORCH_INTERNAL_ASSERT(false, "Unexpected IdMappingMode Type.");
  }
}

static const char* iter_type2string(IterType t) {
  switch (t) {
    case IterType::Iteration:
      return "i";
    case IterType::Reduction:
      return "r";
    case IterType::Broadcast:
      return "b";
    case IterType::Gather:
      return "g";
    case IterType::Stride:
      return "s";
    case IterType::VectorComponent:
      return "v";
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected parallel type");
  }
}

static const char* load_store_type2string(LoadStoreOpType t) {
  switch (t) {
    case LoadStoreOpType::LdMatrix:
      return "LdMatrix";
    case LoadStoreOpType::LdMatrixTranspose:
      return "LdMatrixTranspose";
    case LoadStoreOpType::CpAsync:
      return "CpAsync";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected parallel type");
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
    case supported_switch_pair(DataType::ComplexFloat, DataType::Float):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Float):
      return "(float)std::real";
    case supported_switch_pair(DataType::Index, DataType::Int):
    case supported_switch_pair(DataType::Int32, DataType::Int):
    case supported_switch_pair(DataType::Float, DataType::Int):
    case supported_switch_pair(DataType::Double, DataType::Int):
    case supported_switch_pair(DataType::Bool, DataType::Int):
      return "(int64_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int):
      return "(int64_t)std::real";
    case supported_switch_pair(DataType::Index, DataType::Int32):
    case supported_switch_pair(DataType::Int, DataType::Int32):
    case supported_switch_pair(DataType::Float, DataType::Int32):
    case supported_switch_pair(DataType::Double, DataType::Int32):
    case supported_switch_pair(DataType::Bool, DataType::Int32):
      return "(int32_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int32):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int32):
      return "(int32_t)std::real";
    case supported_switch_pair(DataType::Int, DataType::Index):
    case supported_switch_pair(DataType::Int32, DataType::Index):
    case supported_switch_pair(DataType::Float, DataType::Index):
    case supported_switch_pair(DataType::Double, DataType::Index):
      return "(nvfuser_index_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Index):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Index):
      return "(nvfuser_index_t)std::real";
    case supported_switch_pair(DataType::Index, DataType::Double):
    case supported_switch_pair(DataType::Int, DataType::Double):
    case supported_switch_pair(DataType::Int32, DataType::Double):
    case supported_switch_pair(DataType::Float, DataType::Double):
    case supported_switch_pair(DataType::Bool, DataType::Double):
      return "(double)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Double):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Double):
      return "(double)std::real";
    case supported_switch_pair(DataType::Float, DataType::Bool):
    case supported_switch_pair(DataType::Double, DataType::Bool):
    case supported_switch_pair(DataType::Int32, DataType::Bool):
    case supported_switch_pair(DataType::Int, DataType::Bool):
      return "(bool)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Bool):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Bool):
      return "(bool)std::real";
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
    case supported_switch_pair(DataType::Double, DataType::Half):
      return "__double2half";
    case supported_switch_pair(DataType::Float, DataType::BFloat16):
      return "__float2bfloat";
    case supported_switch_pair(DataType::Half, DataType::Float):
      return "__half2float";
    case supported_switch_pair(DataType::Half, DataType::Double):
      return "__half2double";
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

std::ostream& operator<<(std::ostream& out, const PredicateType ptype) {
  return out << predicate_type2string(ptype);
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

std::ostream& operator<<(std::ostream& out, const RNGOpType rngtype) {
  return out << rng_op_type2string(rngtype);
}

std::ostream& operator<<(std::ostream& out, const ParallelType ptype) {
  return out << stringifyThread(ptype);
}

std::ostream& operator<<(std::ostream& out, const MemoryType mtype) {
  return out << memory_type2string(mtype);
}

std::ostream& operator<<(std::ostream& out, const IdMappingMode immtype) {
  return out << id_map_mode_type2string(immtype);
}

std::ostream& operator<<(
    std::ostream& out,
    const LoadStoreOpType load_store_type) {
  return out << load_store_type2string(load_store_type);
}

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& out,
    const IterType bt) {
  return out << iter_type2string(bt);
}

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const Swizzle2DType& swizzle) {
  switch (swizzle) {
    case Swizzle2DType::NoSwizzle:
      os << "NoSwizzle";
      break;
    case Swizzle2DType::ZShape:
      os << "ZShape";
      break;
    case Swizzle2DType::Transpose:
      os << "Transpose";
      break;
    case Swizzle2DType::XOR:
      os << "Xor";
      break;
    case Swizzle2DType::Scatter:
      os << "Scatter";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined 2D swizzle");
      break;
  }
  return os;
}

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const SwizzleMode& swizzle) {
  switch (swizzle) {
    case SwizzleMode::NoSwizzle:
      os << "NoSwizzle";
      break;
    case SwizzleMode::Loop:
      os << "Loop";
      break;
    case SwizzleMode::Data:
      os << "Data";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined 2D swizzle");
      break;
  }
  return os;
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

c10::optional<std::string> inline_op_str(const RNGOpType rngtype) {
  const char* str = rng_op_type_inline_op2string(rngtype);
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
    case DataType::Double_2:
      return sizeof(double) * 2;
    case DataType::Float_2:
      return sizeof(float) * 2;
    default:
      TORCH_INTERNAL_ASSERT(false, "Size undefined for data type, ", type);
  }
}

size_t dataTypeSize(DataType type, DataType index_type) {
  if (type == DataType::Index) {
    TORCH_INTERNAL_ASSERT(
        index_type == DataType::Int32 || index_type == DataType::Int,
        "Invalid index type of ",
        index_type);
    return dataTypeSize(index_type);
  }
  return dataTypeSize(type);
}

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const DoubleBufferLoopStage loop_stage) {
  switch (loop_stage) {
    case DoubleBufferLoopStage::NotApplicable:
      break;
    case DoubleBufferLoopStage::Prolog:
      os << "{DoubleBufferProlog}";
      break;
    case DoubleBufferLoopStage::Main:
      os << "{DoubleBufferMainLoop}";
      break;
    case DoubleBufferLoopStage::Epilog:
      os << "{DoubleBufferEpilog}";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown double buffer stage");
  }
  return os;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
