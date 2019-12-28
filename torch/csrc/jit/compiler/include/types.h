#pragma once

#include <cstdint>
#include <iostream>

#include "torch/csrc/jit/compiler/include/logging.h"

namespace torch {
namespace jit {
namespace compiler {

using int32 = std::int32_t;

// Switch to PT/Aten dtypes

// Data types for scalar and vector elements.
class Dtype {
 public:
  explicit Dtype(int type) : scalar_type_(type), lanes_(1) {}
  Dtype(int scalar_type, int lanes)
      : scalar_type_(scalar_type), lanes_(lanes) {}
  Dtype(Dtype type, int lanes)
      : scalar_type_(type.scalar_type_), lanes_(lanes) {
    CHECK(type.lanes() == 1);
  }
  int lanes() const {
    return lanes_;
  }
  Dtype scalar_type() const;
  bool operator==(const Dtype& other) const {
    return scalar_type_ == other.scalar_type_ && lanes_ == other.lanes_;
  }
  bool operator!=(const Dtype& other) const {
    return !(*this == other);
  }

 private:
  friend std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);
  int scalar_type_;
  int lanes_; // the width of the element for a vector time
};

extern Dtype kUninitialized;
extern Dtype kInt32;
extern Dtype kFloat32;
extern Dtype kHandle;

template <typename T>
Dtype ToDtype();

template <>
inline Dtype ToDtype<int>() {
  return kInt32;
}

template <>
inline Dtype ToDtype<float>() {
  return kFloat32;
}

inline Dtype BinaryOpDtype(Dtype op1_dtype, Dtype op2_dtype) {
  if (op1_dtype == op2_dtype) {
    return op1_dtype;
  }
  CHECK_EQ(op1_dtype.lanes(), op2_dtype.lanes()) << "vector lengths must match";
  Dtype op1_scalar = op1_dtype.scalar_type();
  Dtype op2_scalar = op2_dtype.scalar_type();

  if (op1_scalar == kInt32 && op2_scalar == kFloat32) {
    return op2_dtype;
  }
  if (op1_scalar == kFloat32 && op2_scalar == kInt32) {
    return op1_dtype;
  }
  LOG(FATAL) << "Invalid dtypes: " << op1_dtype << ", " << op2_dtype;
}

} // namespace compiler
} // namespace jit
} // namespace torch
