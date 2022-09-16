#pragma once

#include <cstdint>
#include <iostream>

#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/exceptions.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using int32 = std::int32_t;

class Dtype;
TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);

using ScalarType = c10::ScalarType;

enum ElementType {
  kAllTypes = 0,
  kIntegralTypes = 1 << 0,
  kFloatingPointTypes = 1 << 1,
  kBoolType = 1 << 2,
  kComplexTypes = 1 << 3,
  kQintTypes = 1 << 4,
  kNonComplexOrQintTypes = kIntegralTypes | kBoolType | kFloatingPointTypes,
};

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
    if (type.lanes() != 1) {
      throw malformed_input("dtype lanes dont match");
    }
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
    return c10::isIntegralType(scalar_type_, true);
  }
  bool is_floating_point() const {
    return c10::isFloatingType(scalar_type_);
  }
  bool is_signed() const {
    return c10::isSignedType(scalar_type_);
  }

  Dtype cloneWithScalarType(ScalarType nt) const {
    return Dtype(nt, lanes_);
  }

 private:
  friend std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);
  ScalarType scalar_type_;
  int lanes_; // the width of the element for a vector time
};

extern TORCH_API Dtype kHandle;

#define NNC_DTYPE_DECLARATION(ctype, name) extern TORCH_API Dtype k##name;

AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, NNC_DTYPE_DECLARATION)
NNC_DTYPE_DECLARATION(c10::quint8, QUInt8);
NNC_DTYPE_DECLARATION(c10::qint8, QInt8);
#undef NNC_DTYPE_DECLARATION

template <typename T>
TORCH_API Dtype ToDtype();

#define NNC_TODTYPE_DECLARATION(ctype, name) \
  template <>                                \
  inline Dtype ToDtype<ctype>() {            \
    return k##name;                          \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, NNC_TODTYPE_DECLARATION)
NNC_TODTYPE_DECLARATION(c10::quint8, QUInt8);
NNC_TODTYPE_DECLARATION(c10::qint8, QInt8);
#undef NNC_TODTYPE_DECLARATION

TORCH_API Dtype ToDtype(ScalarType type);

inline Dtype promoteTypes(Dtype a, Dtype b) {
  if (a.lanes() != b.lanes()) {
    throw malformed_input("promoting types with different lanes");
  }
  return Dtype(
      static_cast<ScalarType>(c10::promoteTypes(
          static_cast<c10::ScalarType>(a.scalar_type()),
          static_cast<c10::ScalarType>(b.scalar_type()))),
      a.lanes());
}

inline Dtype BinaryOpDtype(
    Dtype op1_dtype,
    Dtype op2_dtype,
    ScalarType ret_type = ScalarType::Undefined) {
  if (op1_dtype == op2_dtype) {
    if (ret_type == ScalarType::Undefined) {
      return op1_dtype;
    }

    return ToDtype(ret_type);
  }

  if (op1_dtype.lanes() != op2_dtype.lanes()) {
    throw malformed_input("lanes dont match");
  }
  int lanes = op1_dtype.lanes();

  Dtype resultType = promoteTypes(op1_dtype, op2_dtype);
  if (resultType.scalar_type() == ScalarType::Undefined) {
    throw malformed_input("scalar type doesn't match");
  }

  if (lanes == 1) {
    // Use the fixed scalar Dtypes.
    return ToDtype(resultType.scalar_type());
  }

  return resultType;
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
