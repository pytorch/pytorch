#pragma once

#include <cstdint>
#include <iostream>

#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using int32 = std::int32_t;

class Dtype;
TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);

// Switch to PT/Aten dtypes
enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
  // Undefined must be next to match c10::ScalarType;
  Undefined,
  Handle,
  Uninitialized,
  None,
  NumOptions
};

TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const ScalarType& dtype);

TORCH_API bool is_integral(const ScalarType& type);
TORCH_API bool is_floating_point(const ScalarType& type);

// Data types for scalar and vector elements.
class TORCH_API Dtype {
 public:
  explicit Dtype(int8_t type)
      : scalar_type_(static_cast<ScalarType>(type)), lanes_(1) {}
  explicit Dtype(ScalarType type) : scalar_type_(type), lanes_(1) {}
  Dtype(int8_t type, int lanes)
      : scalar_type_(static_cast<ScalarType>(type)), lanes_(lanes) {}
  Dtype(ScalarType type, int lanes) : scalar_type_(type), lanes_(lanes) {}
  Dtype(Dtype type, int lanes)
      : scalar_type_(type.scalar_type_), lanes_(lanes) {
    CHECK(type.lanes() == 1);
  }
  int lanes() const {
    return lanes_;
  }
  ScalarType scalar_type() const {
    return scalar_type_;
  }
  Dtype scalar_dtype() const;
  bool operator==(const Dtype& other) const {
    return scalar_type_ == other.scalar_type_ && lanes_ == other.lanes_;
  }
  bool operator!=(const Dtype& other) const {
    return !(*this == other);
  }
  int byte_size() const;
  std::string ToCppString() const;

  bool is_integral() const {
    return tensorexpr::is_integral(scalar_type_);
  }
  bool is_floating_point() const {
    return tensorexpr::is_floating_point(scalar_type_);
  }

 private:
  friend std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);
  ScalarType scalar_type_;
  int lanes_; // the width of the element for a vector time
};

extern TORCH_API Dtype kUninitialized;
extern TORCH_API Dtype kHandle;

#define NNC_DTYPE_DECLARATION(ctype, name) extern TORCH_API Dtype k##name;

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, NNC_DTYPE_DECLARATION)
#undef NNC_DTYPE_DECLARATION

template <typename T>
TORCH_API Dtype ToDtype();

#define NNC_TODTYPE_DECLARATION(ctype, name) \
  template <>                                \
  inline Dtype ToDtype<ctype>() {            \
    return k##name;                          \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, NNC_TODTYPE_DECLARATION)
#undef NNC_TODTYPE_DECLARATION

TORCH_API Dtype ToDtype(ScalarType type);

// Call c10 type promotion directly.
inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  return static_cast<ScalarType>(c10::promoteTypes(
      static_cast<c10::ScalarType>(a), static_cast<c10::ScalarType>(b)));
}
inline ScalarType promoteTypes(Dtype a, Dtype b) {
  return static_cast<ScalarType>(c10::promoteTypes(
      static_cast<c10::ScalarType>(a.scalar_type()),
      static_cast<c10::ScalarType>(b.scalar_type())));
}

inline Dtype BinaryOpDtype(
    Dtype op1_dtype,
    Dtype op2_dtype,
    ScalarType ret_type = ScalarType::None) {
  if (op1_dtype == op2_dtype) {
    if (ret_type == ScalarType::None) {
      return op1_dtype;
    }

    return ToDtype(ret_type);
  }

  CHECK_EQ(op1_dtype.lanes(), op2_dtype.lanes()) << "vector lengths must match";
  int lanes = op1_dtype.lanes();

  ScalarType resultType = promoteTypes(op1_dtype, op2_dtype);
  CHECK_NE(resultType, ScalarType::Undefined)
      << "Invalid dtypes: " << op1_dtype << ", " << op2_dtype;

  if (lanes == 1) {
    // Use the fixed scalar Dtypes.
    return ToDtype(resultType);
  }

  return Dtype(resultType, lanes);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

using torch::jit::tensorexpr::Dtype;
std::string to_string(const Dtype& dtype);
using torch::jit::tensorexpr::ScalarType;
std::string to_string(const ScalarType& dtype);

} // namespace std
