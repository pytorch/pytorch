#include "torch/csrc/jit/tensorexpr/types.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <c10/util/Logging.h>

namespace torch {
namespace jit {
namespace tensorexpr {

enum ScalarType {
  kScalarUninitialized,
  kScalarHandle,
  kScalarInt32,
  kScalarFloat32,
};

Dtype Dtype::scalar_type() const {
  switch (static_cast<ScalarType>(scalar_type_)) {
    case kScalarUninitialized:
      return kUninitialized;
    case kScalarHandle:
      return kHandle;
    case kScalarInt32:
      return kInt32;
    case kScalarFloat32:
      return kFloat32;
    default:
      LOG(FATAL) << "invalid scalar type: " << scalar_type_;
      return kUninitialized;
  }
}

TORCH_API Dtype kInt32(kScalarInt32, 1);
TORCH_API Dtype kFloat32(kScalarFloat32, 1);
TORCH_API Dtype kHandle(kScalarHandle, 1);
TORCH_API Dtype kUninitialized(kScalarUninitialized, 1);

TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype) {
  switch (static_cast<ScalarType>(dtype.scalar_type_)) {
    case kScalarUninitialized:
      stream << "uninitialized";
      break;
    case kScalarHandle:
      stream << "handle";
      break;
    case kScalarInt32:
      stream << "int32";
      break;
    case kScalarFloat32:
      stream << "float32";
      break;
    default:
      LOG(FATAL) << "invalid scalar type: " << dtype.scalar_type_;
  }
  if (dtype.lanes() > 1) {
    stream << "x" << dtype.lanes();
    ;
  }
  return stream;
}

int Dtype::byte_size() const {
  int scalar_size = -1;
  switch (scalar_type_) {
    case kScalarInt32:
      scalar_size = sizeof(int32);
      break;
    case kScalarFloat32:
      scalar_size = sizeof(float);
      break;
    default:
      throw std::runtime_error(
          "invalid scalar type; " + std::to_string(scalar_type_));
  }
  return scalar_size * lanes();
}

std::string Dtype::ToCppString() const {
  if (scalar_type_ == kScalarInt32) {
    return "int";
  } else if (scalar_type_ == kScalarFloat32) {
    return "float";
  } else {
    throw std::runtime_error("Invalid dtype: " + std::to_string(scalar_type_));
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

std::string to_string(const Dtype& dtype) {
  std::ostringstream oss;
  oss << dtype;
  return oss.str();
}

} // namespace std
