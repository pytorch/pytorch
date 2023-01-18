#include <torch/csrc/jit/tensorexpr/types.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>

#include <c10/util/Logging.h>

namespace torch::jit::tensorexpr {

Dtype Dtype::scalar_dtype() const {
  return ToDtype(scalar_type_);
}

// NOLINTNEXTLINE
#define DTYPE_DEFINE(_1, n) TORCH_API Dtype k##n(ScalarType::n, 1);

AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DTYPE_DEFINE)
DTYPE_DEFINE(c10::quint8, QUInt8);
DTYPE_DEFINE(c10::qint8, QInt8);

#undef DTYPE_DEFINE

TORCH_API Dtype kHandle(ScalarType::Undefined, 1);

Dtype ToDtype(ScalarType type) {
  switch (type) {
// NOLINTNEXTLINE
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return k##n;
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE)
    TYPE_CASE(c10::quint8, QUInt8);
    TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE

    case ScalarType::Undefined:
      return kHandle;
    default:
      throw unsupported_dtype();
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

int Dtype::byte_size() const {
  int scalar_size = -1;
  switch (scalar_type_) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name)   \
  case ScalarType::Name:        \
    scalar_size = sizeof(Type); \
    break;

    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
    TYPE_CASE(c10::quint8, QUInt8);
    TYPE_CASE(c10::qint8, QInt8);
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
    AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Half:
      return "half";
    case ScalarType::BFloat16:
      return "bfloat16";
    case ScalarType::QInt8:
      return "qint8";
    case ScalarType::QUInt8:
      return "quint8";
    default:
      throw unsupported_dtype();
  }
  return "invalid";
}

} // namespace torch::jit::tensorexpr

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
