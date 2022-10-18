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

class TORCH_CUDA_CU_API IntOrDouble {
  c10::variant<double, int64_t> value_;

 public:
  IntOrDouble(int64_t i) : value_(i) {}
  IntOrDouble(double d) : value_(d) {}
  IntOrDouble(int i) : value_((int64_t)i) {}
  IntOrDouble(size_t i) : value_((int64_t)i) {}
  IntOrDouble() : IntOrDouble(0) {}

  // Avoid using copy constructor of c10::variant as it's
  // deprecated.
  IntOrDouble(const IntOrDouble& other) {
    value_ = other.value_;
  }

  // Explicitly define copy assignment operator as its implicit definition is
  // deprecated
  IntOrDouble& operator=(const IntOrDouble& other) {
    value_ = other.value_;
    return *this;
  }

  bool is_int() const {
    return c10::holds_alternative<int64_t>(value_);
  }

  template <typename T>
  T as() const {
    TORCH_CHECK(
        c10::holds_alternative<T>(value_),
        "The expected dtype and the actual dtype does not match in IntOrDouble");
    return c10::get<T>(value_);
  }

  template <typename T>
  T cast() const;

#define DEFINE_ARITHMETIC_OP(op)                                  \
  IntOrDouble operator op(const IntOrDouble& other) const {       \
    switch ((int)is_int() << 1 | (int)other.is_int()) {           \
      case 0b00:                                                  \
        return IntOrDouble(as<double>() op other.as<double>());   \
      case 0b01:                                                  \
        return IntOrDouble(as<double>() op other.as<int64_t>());  \
      case 0b10:                                                  \
        return IntOrDouble(as<int64_t>() op other.as<double>());  \
      case 0b11:                                                  \
        return IntOrDouble(as<int64_t>() op other.as<int64_t>()); \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }                                                               \
  template <typename T>                                           \
  IntOrDouble operator op(T other) const {                        \
    if (is_int()) {                                               \
      return IntOrDouble(as<int64_t>() op other);                 \
    }                                                             \
    return IntOrDouble(as<double>() op other);                    \
  }

  DEFINE_ARITHMETIC_OP(+)
  DEFINE_ARITHMETIC_OP(-)
  DEFINE_ARITHMETIC_OP(*)
  DEFINE_ARITHMETIC_OP(/)
  DEFINE_ARITHMETIC_OP(&&)

#undef DEFINE_ARITHMETIC_OP

#define DEFINE_ASSIGN_OP(assign, op)                                      \
  IntOrDouble& operator assign(const IntOrDouble& other) {                \
    switch ((int)is_int() << 1 | (int)other.is_int()) {                   \
      case 0b00:                                                          \
        return *this = IntOrDouble(as<double>() op other.as<double>());   \
      case 0b01:                                                          \
        return *this = IntOrDouble(as<double>() op other.as<int64_t>());  \
      case 0b10:                                                          \
        return *this = IntOrDouble(as<int64_t>() op other.as<double>());  \
      case 0b11:                                                          \
        return *this = IntOrDouble(as<int64_t>() op other.as<int64_t>()); \
    }                                                                     \
    TORCH_INTERNAL_ASSERT(false);                                         \
  }                                                                       \
  template <typename T>                                                   \
  IntOrDouble& operator assign(T other) {                                 \
    if (is_int()) {                                                       \
      return *this = IntOrDouble(as<int64_t>() op other);                 \
    }                                                                     \
    return *this = IntOrDouble(as<double>() op other);                    \
  }

  DEFINE_ASSIGN_OP(+=, +)
  DEFINE_ASSIGN_OP(-=, -)
  DEFINE_ASSIGN_OP(*=, *)
  DEFINE_ASSIGN_OP(/=, /)

#undef DEFINE_ASSIGN_OP

  IntOrDouble operator%(const IntOrDouble& other) const {
    if (is_int() && other.is_int()) {
      return IntOrDouble(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  IntOrDouble operator%(int64_t other) const {
    if (is_int()) {
      return IntOrDouble(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  IntOrDouble& operator%=(const IntOrDouble& other) {
    if (is_int() && other.is_int()) {
      return *this = IntOrDouble(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  IntOrDouble& operator%=(int64_t other) {
    if (is_int()) {
      return *this = IntOrDouble(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }

#define DEFINE_COMPARE_OP(op)                           \
  bool operator op(const IntOrDouble& other) const {    \
    switch ((int)is_int() << 1 | (int)other.is_int()) { \
      case 0b00:                                        \
        return as<double>() op other.as<double>();      \
      case 0b01:                                        \
        return as<double>() op other.as<int64_t>();     \
      case 0b10:                                        \
        return as<int64_t>() op other.as<double>();     \
      case 0b11:                                        \
        return as<int64_t>() op other.as<int64_t>();    \
    }                                                   \
    TORCH_INTERNAL_ASSERT(false);                       \
  }                                                     \
  bool operator op(double other) {                      \
    if (is_int()) {                                     \
      return as<int64_t>() op other;                    \
    }                                                   \
    return as<double>() op other;                       \
  }                                                     \
  bool operator op(int64_t other) {                     \
    if (is_int()) {                                     \
      return as<int64_t>() op other;                    \
    }                                                   \
    return as<double>() op other;                       \
  }                                                     \
  bool operator op(int other) {                         \
    if (is_int()) {                                     \
      return as<int64_t>() op other;                    \
    }                                                   \
    return as<double>() op other;                       \
  }

  DEFINE_COMPARE_OP(>)
  DEFINE_COMPARE_OP(>=)
  DEFINE_COMPARE_OP(<)
  DEFINE_COMPARE_OP(<=)
  DEFINE_COMPARE_OP(==)
  DEFINE_COMPARE_OP(!=)

#undef DEFINE_COMPARE_OP

  IntOrDouble operator-() const {
    if (is_int()) {
      return IntOrDouble(-as<int64_t>());
    }
    return IntOrDouble(-as<double>());
  }

  explicit operator double() const;
  explicit operator int64_t() const;
  explicit operator size_t() const;
  explicit operator int() const;
};

#define DEFINE_ARITHMETIC_OP(op)                           \
  template <typename T>                                    \
  inline IntOrDouble operator op(T lhs, IntOrDouble rhs) { \
    if (rhs.is_int()) {                                    \
      return IntOrDouble(lhs op rhs.as<int64_t>());        \
    }                                                      \
    return IntOrDouble(lhs op rhs.as<double>());           \
  }

DEFINE_ARITHMETIC_OP(+)
DEFINE_ARITHMETIC_OP(-)
DEFINE_ARITHMETIC_OP(*)
DEFINE_ARITHMETIC_OP(/)

#undef DEFINE_ARITHMETIC_OP

template <>
inline double IntOrDouble::cast<double>() const {
  if (is_int()) {
    return (double)as<int64_t>();
  }
  return as<double>();
}

template <>
inline int64_t IntOrDouble::cast<int64_t>() const {
  if (!is_int()) {
    return (int64_t)as<double>();
  }
  return as<int64_t>();
}

inline IntOrDouble::operator double() const {
  return as<double>();
}

inline IntOrDouble::operator int64_t() const {
  return as<int64_t>();
}

inline IntOrDouble::operator size_t() const {
  return as<int64_t>();
}

inline IntOrDouble::operator int() const {
  return as<int64_t>();
}

#define DEFINE_EQ_OP(op)                                         \
  inline bool operator op(double lhs, const IntOrDouble& rhs) {  \
    if (rhs.is_int()) {                                          \
      return false;                                              \
    }                                                            \
    return lhs op rhs.as<double>();                              \
  }                                                              \
                                                                 \
  inline bool operator op(int64_t lhs, const IntOrDouble& rhs) { \
    if (rhs.is_int()) {                                          \
      return lhs op rhs.as<int64_t>();                           \
    }                                                            \
    return false;                                                \
  }                                                              \
                                                                 \
  inline bool operator op(int lhs, const IntOrDouble& rhs) {     \
    return operator op((int64_t)lhs, rhs);                       \
  }

DEFINE_EQ_OP(==)
DEFINE_EQ_OP(!=)

#undef DEFINE_EQ_OP

inline std::ostream& operator<<(std::ostream& os, const IntOrDouble& val) {
  if (val.is_int()) {
    return os << val.as<int64_t>();
  }
  return os << val.as<double>();
}

namespace IntOrDouble_functions {

inline IntOrDouble ceildiv(const IntOrDouble& a, const IntOrDouble& b) {
  if (a.is_int() && b.is_int()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return (aa + bb - 1) / bb;
    } else {
      return (aa + bb + 1) / bb;
    }
  }
  return std::ceil((a / b).as<double>());
}

inline IntOrDouble max(const IntOrDouble& a, const IntOrDouble& b) {
  if (a.is_int() && b.is_int()) {
    return std::max(a.as<int64_t>(), b.as<int64_t>());
  }
  return (a > b ? a : b).cast<double>();
}

inline IntOrDouble min(const IntOrDouble& a, const IntOrDouble& b) {
  if (a.is_int() && b.is_int()) {
    return std::min(a.as<int64_t>(), b.as<int64_t>());
  }
  return (a < b ? a : b).cast<double>();
}

} // namespace IntOrDouble_functions

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
