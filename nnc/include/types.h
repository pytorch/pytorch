#ifndef NNC_INCLUDE_DTYPES_H_INCLUDED__
#define NNC_INCLUDE_DTYPES_H_INCLUDED__

#include <cstdint>

#include "logging.h"

namespace nnc {

using int32 = std::int32_t;

// Switch to PT/Aten dtypes
enum Dtype {
  kUninitialized,
  kInt32,
  kFloat32,
};

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
  if (op1_dtype == kInt32 && op2_dtype == kFloat32) {
    return kFloat32;
  }
  if (op1_dtype == kFloat32 && op2_dtype == kInt32) {
    return kFloat32;
  }
  LOG(FATAL) << "Invalid dtypes: " << op1_dtype << ", " << op2_dtype;
}

class Scalar {
 public:
  Scalar() : dtype_(kInt32) { i32_value = 0; }

  Scalar(int v) : dtype_(kInt32) { i32_value = v; }

  Scalar(float v) : dtype_(kFloat32) { f32_value = v; }

  template <typename T>
  T as() const;

  Dtype dtype() const { return dtype_; }

 private:
  enum Dtype dtype_;
  union {
    int32 i32_value;
    float f32_value;
  };
};

template <>
inline int Scalar::as<int>() const {
  CHECK_EQ(dtype_, kInt32) << "invalid dtype";
  return i32_value;
}

template <>
inline float Scalar::as<float>() const {
  CHECK_EQ(dtype_, kFloat32) << "invalid dtype";
  return f32_value;
}

}  // namespace nnc

#endif  //  NNC_INCLUDE_DTYPES_H_INCLUDED__
