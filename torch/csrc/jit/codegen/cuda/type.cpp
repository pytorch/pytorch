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

template <typename T>
struct _enum_class_hash {
  size_t operator()(T v) const {
    return static_cast<size_t>(v);
  }
};
template <typename KeyType, typename ValType>
using _enum_unordered_map =
    std::unordered_map<KeyType, ValType, _enum_class_hash<KeyType>>;
static _enum_unordered_map<DataType, std::string> data_type_string_map{
    {DataType::Bool, "bool"},
    {DataType::Float, "float"},
    {DataType::Half, "__half"},
    {DataType::Int, "size_t"}};
static _enum_unordered_map<ValType, std::string> val_type_string_map{
    {ValType::TensorIndex, "TensorIndex"},
    {ValType::TensorView, "TensorView"},
    {ValType::TensorDomain, "TensorDomain"},
    {ValType::IterDomain, "IterDomain"},
    {ValType::Scalar, "Scalar"},
    {ValType::NamedScalar, "NamedScalar"}};

static _enum_unordered_map<ExprType, std::string> expr_type_string_map{
    {ExprType::UnaryOp, "UnaryOp"},
    {ExprType::BinaryOp, "BinaryOp"},
    {ExprType::TernaryOp, "TernaryOp"},
    {ExprType::ForLoop, "ForLoop"},
    {ExprType::IfThenElse, "IfThenElse"},
    {ExprType::Allocate, "Allocate"},
    {ExprType::Split, "Split"},
    {ExprType::Merge, "Merge"},
    {ExprType::Reorder, "Reorder"}};
static _enum_unordered_map<UnaryOpType, std::string> unary_op_type_string_map{
    {UnaryOpType::Abs, "fabs"},
    {UnaryOpType::Acos, "acosf"},
    {UnaryOpType::Asin, "asinf"},
    {UnaryOpType::Atan, "atanf"},
    {UnaryOpType::Atanh, "atanhf"},
    {UnaryOpType::Cast, "cast"},
    {UnaryOpType::Ceil, "ceilf"},
    {UnaryOpType::Cos, "cosf"},
    {UnaryOpType::Cosh, "coshf"},
    {UnaryOpType::Exp, "expf"},
    {UnaryOpType::Expm1, "expm1f"},
    {UnaryOpType::Erf, "erff"},
    {UnaryOpType::Erfc, "erfcf"},
    {UnaryOpType::Floor, "floorf"},
    {UnaryOpType::Frac, "frac"},
    {UnaryOpType::Gelu, "gelu"},
    {UnaryOpType::Lgamma, "lgammaf"},
    {UnaryOpType::Log, "logf"},
    {UnaryOpType::Log10, "log10f"},
    {UnaryOpType::Log1p, "log1pf"},
    {UnaryOpType::Log2, "log2f"},
    {UnaryOpType::Neg, "neg"},
    {UnaryOpType::RandLike, "randLike"},
    {UnaryOpType::Reciprocal, "reciprocal"},
    {UnaryOpType::Relu, "relu"},
    {UnaryOpType::Rsqrt, "rsqrtf"},
    {UnaryOpType::Round, "roundf"},
    {UnaryOpType::Set, "set"},
    {UnaryOpType::Sigmoid, "sigmoid"},
    {UnaryOpType::Sin, "sinf"},
    {UnaryOpType::Sinh, "sinhf"},
    {UnaryOpType::Sqrt, "sqrtf"},
    {UnaryOpType::Tan, "tanf"},
    {UnaryOpType::Tanh, "tanhf"},
    {UnaryOpType::Trunc, "truncf"}};
static _enum_unordered_map<UnaryOpType, std::string>
    unary_op_type_inline_op_string_map{{UnaryOpType::Neg, "-"},
                                       {UnaryOpType::Set, ""}};
static _enum_unordered_map<BinaryOpType, std::string> binary_op_type_string_map{
    {BinaryOpType::Add, "add"},
    {BinaryOpType::Atan2, "atan2f"},
    {BinaryOpType::Div, "div"},
    {BinaryOpType::Fmod, "fmodf"},
    {BinaryOpType::Max, "fmaxf"},
    {BinaryOpType::Min, "fminf"},
    {BinaryOpType::Mul, "mul"},
    {BinaryOpType::Pow, "powf"},
    {BinaryOpType::Remainder, "remainder"},
    {BinaryOpType::Sub, "sub"},
    //{BinaryOpType::TypeAs,

    // Logical Ops
    {BinaryOpType::Mod, "mod"},
    {BinaryOpType::CeilDiv, "ceilDiv"},
    {BinaryOpType::And, "and"},
    {BinaryOpType::Eq, "equal"},
    {BinaryOpType::GE, "greaterThanOrEqual"},
    {BinaryOpType::GT, "greaterThan"},
    {BinaryOpType::LE, "lessThanOrEqual"},
    {BinaryOpType::LT, "lessThan"},
    {BinaryOpType::NE, "notEqual"}};

static _enum_unordered_map<BinaryOpType, std::string>
    binary_op_type_inline_op_string_map{{BinaryOpType::Add, "+"},
                                        {BinaryOpType::Div, "/"},
                                        {BinaryOpType::Mod, "%"},
                                        {BinaryOpType::Mul, "*"},
                                        {BinaryOpType::Sub, "-"},

                                        // Logical Ops
                                        {BinaryOpType::And, "&&"},
                                        {BinaryOpType::Eq, "=="},
                                        {BinaryOpType::GE, ">="},
                                        {BinaryOpType::GT, ">"},
                                        {BinaryOpType::LE, "<="},
                                        {BinaryOpType::LT, "<"},
                                        {BinaryOpType::NE, "!="}};
static _enum_unordered_map<TernaryOpType, std::string>
    ternary_op_type_string_map{{TernaryOpType::Clamp, "clamp"},
                               {TernaryOpType::Threshold, "threshold"},
                               {TernaryOpType::Where, "where"}};

static _enum_unordered_map<ParallelType, std::string> parallel_type_string_map{
    {ParallelType::BIDz, "blockIdx.z"},
    {ParallelType::BIDy, "blockIdx.y"},
    {ParallelType::BIDx, "blockIdx.x"},
    {ParallelType::TIDz, "threadIdx.z"},
    {ParallelType::TIDy, "threadIdx.y"},
    {ParallelType::TIDx, "threadIdx.x"},
    {ParallelType::Vectorize, "Vectorize"},
    {ParallelType::Unroll, "Unroll"},
    {ParallelType::Serial, "Serial"}};

static _enum_unordered_map<MemoryType, std::string> memory_type_string_map{
    {MemoryType::Local, "register"},
    {MemoryType::Shared, "shared"},
    {MemoryType::Global, "global"}};

static _enum_unordered_map<at::ScalarType, DataType> at_type_map{
    {at::ScalarType::Bool, DataType::Bool},
    {at::ScalarType::Float, DataType::Float},
    {at::ScalarType::Half, DataType::Half},
    {at::ScalarType::Int, DataType::Int}};

static _enum_unordered_map<ParallelType, std::string> thread_size_string_map{
    {ParallelType::BIDz, "gridDim.z"},
    {ParallelType::BIDy, "gridDim.y"},
    {ParallelType::BIDx, "gridDim.x"},
    {ParallelType::TIDz, "blockDim.z"},
    {ParallelType::TIDy, "blockDim.y"},
    {ParallelType::TIDx, "blockDim.x"}};

static std::unordered_set<BinaryOpType, _enum_class_hash<BinaryOpType>>
    logical_binary_ops{BinaryOpType::And,
                       BinaryOpType::Eq,
                       BinaryOpType::GE,
                       BinaryOpType::GT,
                       BinaryOpType::LE,
                       BinaryOpType::LT,
                       BinaryOpType::NE};

template <typename T>
struct _enum_pair_hash {
  size_t operator()(std::pair<T, T> p) const {
    return static_cast<size_t>(p.first) ^ static_cast<size_t>(p.second);
  }
};

template <typename KeyType, typename ValType>
using _enum_pair_unordered_map = std::unordered_map<
    std::pair<KeyType, KeyType>,
    ValType,
    _enum_pair_hash<KeyType>>;

static _enum_pair_unordered_map<DataType, std::string> supported_casts{
    {{DataType::Float, DataType::Half}, "__float2half"},
    {{DataType::Half, DataType::Float}, "__half2float"}};

bool is_logical_op(const BinaryOpType& bot) {
  return logical_binary_ops.count(bot) > 0;
}

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  TORCH_INTERNAL_ASSERT(
      at_type_map.count(scalar_type) != 0, "No string found for scalar type.");
  return at_type_map[scalar_type];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ValType vtype) {
  TORCH_INTERNAL_ASSERT(
      val_type_string_map.count(vtype) != 0, "No string found for val type.");
  return out << val_type_string_map[vtype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const DataType dtype) {
  TORCH_INTERNAL_ASSERT(
      data_type_string_map.count(dtype) != 0, "No string found for data type.");
  return out << data_type_string_map[dtype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ExprType etype) {
  TORCH_INTERNAL_ASSERT(
      expr_type_string_map.count(etype) != 0, "No string found for expr type.");
  return out << expr_type_string_map[etype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const UnaryOpType uotype) {
  TORCH_INTERNAL_ASSERT(
      unary_op_type_string_map.count(uotype) != 0,
      "No string found for UnaryOp type.");
  return out << unary_op_type_string_map[uotype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const BinaryOpType botype) {
  TORCH_INTERNAL_ASSERT(
      binary_op_type_string_map.count(botype) != 0,
      "No string found for BinaryOp type.");
  return out << binary_op_type_string_map[botype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const TernaryOpType totype) {
  TORCH_INTERNAL_ASSERT(
      ternary_op_type_string_map.count(totype) != 0,
      "No string found for TernaryOp type.");
  return out << ternary_op_type_string_map[totype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ParallelType ptype) {
  TORCH_INTERNAL_ASSERT(
      parallel_type_string_map.count(ptype) != 0,
      "No string found for provided parallel type.");
  return out << parallel_type_string_map[ptype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const MemoryType mtype) {
  TORCH_INTERNAL_ASSERT(
      memory_type_string_map.count(mtype) != 0,
      "No string found for provided memory type.");
  return out << memory_type_string_map[mtype];
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const UnaryOpType uotype) {
  if (unary_op_type_inline_op_string_map.find(uotype) ==
      unary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return unary_op_type_inline_op_string_map[uotype];
  }
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const BinaryOpType botype) {
  if (binary_op_type_inline_op_string_map.find(botype) ==
      binary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return binary_op_type_inline_op_string_map[botype];
  }
}

std::string stringifyThreadSize(const ParallelType ptype) {
  TORCH_INTERNAL_ASSERT(
      thread_size_string_map.find(ptype) != thread_size_string_map.end(),
      "Could not find size of the thread type ",
      ptype);
  return thread_size_string_map[ptype];
}

TORCH_CUDA_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>& cast) {
  if (supported_casts.count(cast) == 0) {
    return c10::nullopt;
  } else {
    return supported_casts[cast];
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
