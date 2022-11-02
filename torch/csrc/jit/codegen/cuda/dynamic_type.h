#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/variant.h>
#include <cmath>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API EvaluatorValue {
  c10::variant<double, int64_t, bool> value_;

 public:
  explicit EvaluatorValue(int64_t i) : value_(i) {}
  explicit EvaluatorValue(double d) : value_(d) {}
  explicit EvaluatorValue(bool b) : value_(b) {}
  explicit EvaluatorValue(int i) : value_((int64_t)i) {}
  explicit EvaluatorValue(size_t i) : value_((int64_t)i) {}
  EvaluatorValue() : EvaluatorValue(0) {}

  // Avoid using copy constructor of c10::variant as it's
  // deprecated.
  EvaluatorValue(const EvaluatorValue& other) {
    value_ = other.value_;
  }

  // Explicitly define copy assignment operator as its implicit definition is
  // deprecated
  EvaluatorValue& operator=(const EvaluatorValue& other) {
    value_ = other.value_;
    return *this;
  }

  bool isInt() const {
    return c10::holds_alternative<int64_t>(value_);
  }

  bool isDouble() const {
    return c10::holds_alternative<double>(value_);
  }

  bool isBool() const {
    return c10::holds_alternative<bool>(value_);
  }

  template <typename T>
  T as() const {
    TORCH_CHECK(
        c10::holds_alternative<T>(value_),
        "The expected dtype and the actual dtype does not match in EvaluatorValue");
    return c10::get<T>(value_);
  }

  template <typename T>
  T cast() const {
    if (isInt()) {
      return (T)as<int64_t>();
    }
    if (isBool()) {
      return (T)as<bool>();
    }
    if (isDouble()) {
      return (T)as<double>();
    }
    TORCH_INTERNAL_ASSERT(false);
  }

#define DEFINE_ARITHMETIC_OP(op)                                  \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    if (isInt()) {                                                \
      return EvaluatorValue(as<int64_t>() op other);              \
    }                                                             \
    if (isDouble()) {                                             \
      return EvaluatorValue(as<double>() op other);               \
    }                                                             \
    if (isBool()) {                                               \
      return EvaluatorValue(as<bool>() op other);                 \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    if (other.isInt()) {                                          \
      return operator op(other.as<int64_t>());                    \
    }                                                             \
    if (other.isDouble()) {                                       \
      return operator op(other.as<double>());                     \
    }                                                             \
    if (other.isBool()) {                                         \
      return operator op(other.as<bool>());                       \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }

  DEFINE_ARITHMETIC_OP(+)
  DEFINE_ARITHMETIC_OP(-)
  DEFINE_ARITHMETIC_OP(*)
  DEFINE_ARITHMETIC_OP(/)
  DEFINE_ARITHMETIC_OP(>)
  DEFINE_ARITHMETIC_OP(>=)
  DEFINE_ARITHMETIC_OP(<)
  DEFINE_ARITHMETIC_OP(<=)
  DEFINE_ARITHMETIC_OP(==)
  DEFINE_ARITHMETIC_OP(!=)

#undef DEFINE_ARITHMETIC_OP

#define DEFINE_BITWISE_OP(op)                                     \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    if (isInt()) {                                                \
      return EvaluatorValue(as<int64_t>() op other);              \
    }                                                             \
    if (isBool()) {                                               \
      return EvaluatorValue(as<bool>() op other);                 \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    if (other.isInt()) {                                          \
      return operator op(other.as<int64_t>());                    \
    }                                                             \
    if (other.isBool()) {                                         \
      return operator op(other.as<bool>());                       \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }

  DEFINE_BITWISE_OP(|)
  DEFINE_BITWISE_OP(^)
  DEFINE_BITWISE_OP(&)

#undef DEFINE_BITWISE_OP

#define DEFINE_LOGICAL_OP(op)                                     \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    return EvaluatorValue(cast<bool>() op other);                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    return operator op(other.cast<bool>());                       \
  }

  DEFINE_LOGICAL_OP(||)
  DEFINE_LOGICAL_OP(&&)

#undef DEFINE_LOGICAL_OP

#define DEFINE_ASSIGN_OP(assign, op)                             \
  EvaluatorValue& operator assign(const EvaluatorValue& other) { \
    *this = *this op other;                                      \
    return *this;                                                \
  }                                                              \
  template <typename T>                                          \
  EvaluatorValue& operator assign(T other) {                     \
    *this = *this op other;                                      \
    return *this;                                                \
  }

  DEFINE_ASSIGN_OP(+=, +)
  DEFINE_ASSIGN_OP(-=, -)
  DEFINE_ASSIGN_OP(*=, *)
  DEFINE_ASSIGN_OP(/=, /)
  DEFINE_ASSIGN_OP(&=, &)
  DEFINE_ASSIGN_OP(|=, |)
  DEFINE_ASSIGN_OP(^=, ^)

#undef DEFINE_ASSIGN_OP

  EvaluatorValue operator%(const EvaluatorValue& other) const {
    if (isInt() && other.isInt()) {
      return EvaluatorValue(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue operator%(int64_t other) const {
    if (isInt()) {
      return EvaluatorValue(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue& operator%=(const EvaluatorValue& other) {
    if (isInt() && other.isInt()) {
      return *this = EvaluatorValue(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue& operator%=(int64_t other) {
    if (isInt()) {
      return *this = EvaluatorValue(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }

  EvaluatorValue operator-() const {
    if (isInt()) {
      return EvaluatorValue(-as<int64_t>());
    }
    if (isDouble()) {
      return EvaluatorValue(-as<double>());
    }
    if (isBool()) {
      return EvaluatorValue(-as<bool>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }

  explicit operator double() const;
  explicit operator int64_t() const;
  explicit operator size_t() const;
  explicit operator int() const;
  explicit operator bool() const;
}; // namespace cuda

#define DEFINE_ARITHMETIC_OP(op)                                 \
  template <typename T>                                          \
  inline EvaluatorValue operator op(T lhs, EvaluatorValue rhs) { \
    return EvaluatorValue(lhs) op rhs;                           \
  }

DEFINE_ARITHMETIC_OP(+)
DEFINE_ARITHMETIC_OP(-)
DEFINE_ARITHMETIC_OP(*)
DEFINE_ARITHMETIC_OP(/)
DEFINE_ARITHMETIC_OP(&&)
DEFINE_ARITHMETIC_OP(&)
DEFINE_ARITHMETIC_OP(||)
DEFINE_ARITHMETIC_OP(|)
DEFINE_ARITHMETIC_OP(^)
DEFINE_ARITHMETIC_OP(>)
DEFINE_ARITHMETIC_OP(>=)
DEFINE_ARITHMETIC_OP(<)
DEFINE_ARITHMETIC_OP(<=)
DEFINE_ARITHMETIC_OP(==)
DEFINE_ARITHMETIC_OP(!=)

#undef DEFINE_ARITHMETIC_OP

inline EvaluatorValue::operator double() const {
  return as<double>();
}

inline EvaluatorValue::operator int64_t() const {
  return as<int64_t>();
}

inline EvaluatorValue::operator size_t() const {
  return as<int64_t>();
}

inline EvaluatorValue::operator int() const {
  return as<int64_t>();
}

inline EvaluatorValue::operator bool() const {
  return as<bool>();
}

#undef DEFINE_EQ_OP

inline std::ostream& operator<<(std::ostream& os, const EvaluatorValue& val) {
  if (val.isInt()) {
    return os << val.as<int64_t>();
  }
  if (val.isBool()) {
    return os << val.as<bool>();
  }
  if (val.isDouble()) {
    return os << val.as<double>();
  }
  TORCH_INTERNAL_ASSERT(false);
}

namespace EvaluatorValue_functions {

inline EvaluatorValue ceildiv(
    const EvaluatorValue& a,
    const EvaluatorValue& b) {
  if (a.isInt() && b.isInt()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return EvaluatorValue((aa + bb - 1) / bb);
    } else {
      return EvaluatorValue((aa + bb + 1) / bb);
    }
  }
  return EvaluatorValue(std::ceil((a / b).as<double>()));
}

inline EvaluatorValue max(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a > b).as<bool>() ? a : b);
}

inline EvaluatorValue min(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a < b).as<bool>() ? a : b);
}

inline EvaluatorValue abs(const EvaluatorValue& a) {
  if (a.isInt()) {
    return EvaluatorValue(std::abs(a.as<int64_t>()));
  }
  if (a.isDouble()) {
    return EvaluatorValue(std::abs(a.as<double>()));
  }
  if (a.isBool()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace EvaluatorValue_functions

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
