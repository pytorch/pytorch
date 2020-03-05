#include "torch/csrc/jit/tensorexpr/types.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <c10/util/Logging.h>

namespace torch {
namespace jit {
namespace tensorexpr {

bool is_integral(const ScalarType& type) {
  switch (type) {
    case ScalarType::Byte:
    case ScalarType::Char:
    case ScalarType::Short:
    case ScalarType::Int:
    case ScalarType::Long:
      return true;
    default:
      return false;
  }

  return false;
}

bool is_floating_point(const ScalarType& type) {
  switch (type) {
    case ScalarType::Half:
    case ScalarType::Float:
    case ScalarType::Double:
      return true;
    default:
      return false;
  }

  return false;
}

Dtype Dtype::scalar_dtype() const {
  return ToDtype(scalar_type_);
}

// NOLINTNEXTLINE
#define DTYPE_DEFINE(_1, n) TORCH_API Dtype k##n(ScalarType::n, 1);

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, DTYPE_DEFINE)

#undef DTYPE_DEFINE

TORCH_API Dtype kHandle(ScalarType::Handle, 1);
TORCH_API Dtype kUninitialized(ScalarType::Uninitialized, 1);

Dtype ToDtype(ScalarType type) {
  switch (type) {
// NOLINTNEXTLINE
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return k##n;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE)
#undef TYPE_CASE

    case ScalarType::Handle:
      return kHandle;
    case ScalarType::Uninitialized:
      return kUninitialized;
    default:
      LOG(FATAL) << "invalid scalar type: " << type;
      return kUninitialized;
  }
}

TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype) {
  stream << dtype.scalar_type_;
  if (dtype.lanes() > 1) {
    stream << "x" << dtype.lanes();
    ;
  }
  return stream;
}

TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const ScalarType& type) {
  switch (type) {
// NOLINTNEXTLINE
#define TYPE_CASE(ttt, Name) \
  case ScalarType::Name:     \
    stream << #ttt;          \
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE

    case ScalarType::Undefined:
      stream << "Undefined";
      break;
    case ScalarType::Handle:
      stream << "Handle";
      break;
    case ScalarType::Uninitialized:
      stream << "Uninitialized";
      break;
    case ScalarType::None:
      stream << "None";
      break;
    default:
      LOG(FATAL) << "invalid scalar type: " << (int)type;
  }
  return stream;
}

int Dtype::byte_size() const {
  int scalar_size = -1;
  switch (scalar_type_) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name)   \
  case ScalarType::Name:        \
    scalar_size = sizeof(Type); \
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw std::runtime_error(
          "invalid scalar type; " + std::to_string(scalar_type_));
  }
  return scalar_size * lanes();
}

std::string Dtype::ToCppString() const {
  switch (scalar_type_) {
// NOLINTNEXTLINE
#define TYPE_CASE(t, n) \
  case ScalarType::n:   \
    return #t;
    AT_FORALL_SCALAR_TYPES_AND(Bool, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::Half:
      return "half";
    default:
      LOG(FATAL) << "ToCppString Invalid dtype: "
                 << std::to_string(scalar_type_);
  }
  return "invalid";
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

std::string to_string(const ScalarType& type) {
  std::ostringstream oss;
  oss << type;
  return oss.str();
}

} // namespace std
