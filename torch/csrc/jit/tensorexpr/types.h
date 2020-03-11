#pragma once

#include <cstdint>
#include <iostream>

#include <c10/util/Logging.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using int32 = std::int32_t;

class Dtype;
TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);

// Switch to PT/Aten dtypes

// Data types for scalar and vector elements.
class TORCH_API Dtype {
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
  int byte_size() const;
  std::string ToCppString() const;

 private:
  friend std::ostream& operator<<(std::ostream& stream, const Dtype& dtype);
  int scalar_type_;
  int lanes_; // the width of the element for a vector time
};

extern TORCH_API Dtype kUninitialized;
extern TORCH_API Dtype kInt32;
extern TORCH_API Dtype kFloat32;
extern TORCH_API Dtype kHandle;

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

// Optional return type in case
// the binary Op is a CompareSelect Op
enum ReturnType {
  knone,
  kint32,
  kfloat32,
};

inline Dtype BinaryOpDtype(
    Dtype op1_dtype,
    Dtype op2_dtype,
    ReturnType ret_type = ReturnType::knone) {
  if (op1_dtype == op2_dtype) {
    switch (ret_type) {
      case ReturnType::knone:
        return op1_dtype;
      case ReturnType::kint32:
        return ToDtype<int>();
      case ReturnType::kfloat32:
        return ToDtype<float>();
      default:
        throw std::runtime_error("invalid operator return type");
    }
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
  return op1_dtype;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

using torch::jit::tensorexpr::Dtype;
std::string to_string(const Dtype& dtype);

} // namespace std
