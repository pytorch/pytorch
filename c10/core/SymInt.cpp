#include <c10/core/LargeNegativeIntSymNodeImpl.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <array>
#include <functional>
#include <utility>

namespace c10 {

// Precondition: data_ has a large negative number that should be
// treated as a constant.  It is NOT a valid pointer.  In other words,
// SymInt has temporarily violated invariants
// Postcondition: invariants on SymInt are fixed
void SymInt::promote_to_negative() {
  auto s =
      SymInt(SymNode(c10::make_intrusive<LargeNegativeIntSymNodeImpl>(data_)));
  // Similar to move operator=, but do NOT release data_
  data_ = s.data_;
  s.data_ = 0;
}

SymNode SymInt::toSymNode() const {
  TORCH_CHECK(is_heap_allocated());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymInt::SymInt(SymNode sin_sp) {
  TORCH_CHECK(sin_sp->is_int());
  auto ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(static_cast<void*>(sin_sp.release())));
  auto rep = (ptr & ~MASK) | IS_SYM;
  data_ = static_cast<int64_t>(rep);
}

bool SymInt::has_hint() const {
  if (!is_heap_allocated()) {
    return true;
  }
  return toSymNodeImplUnowned()->has_hint();
}

#define DEFINE_BINARY(API, OP, METHOD, RET)                          \
  RET SymInt::API(const SymInt& sci) const {                         \
    if (auto ma = maybe_as_int()) {                                  \
      if (auto mb = sci.maybe_as_int()) {                            \
        return RET(OP(*ma, *mb));                                    \
      } else {                                                       \
        auto b = sci.toSymNode();                                    \
        return RET(b->wrap_int(*ma)->METHOD(b));                     \
      }                                                              \
    } else {                                                         \
      if (auto mb = sci.maybe_as_int()) {                            \
        auto a = toSymNodeImplUnowned();                             \
        return RET(a->METHOD(a->wrap_int(*mb)));                     \
      } else {                                                       \
        return RET(toSymNodeImplUnowned()->METHOD(sci.toSymNode())); \
      }                                                              \
    }                                                                \
  }

// clang-format off
DEFINE_BINARY(operator+, std::plus<>(), add, SymInt)
DEFINE_BINARY(operator-, std::minus<>(), sub, SymInt)
DEFINE_BINARY(operator*, std::multiplies<>(), mul, SymInt)
DEFINE_BINARY(operator/, std::divides<>(), floordiv, SymInt)
DEFINE_BINARY(operator%, std::modulus<>(), mod, SymInt)
DEFINE_BINARY(sym_eq, std::equal_to<>(), eq, SymBool)
DEFINE_BINARY(sym_ne, std::not_equal_to<>(), ne, SymBool)
DEFINE_BINARY(sym_lt, std::less<>(), lt, SymBool)
DEFINE_BINARY(sym_le, std::less_equal<>(), le, SymBool)
DEFINE_BINARY(sym_gt, std::greater<>(), gt, SymBool)
DEFINE_BINARY(sym_ge, std::greater_equal<>(), ge, SymBool)
DEFINE_BINARY(min, std::min, sym_min, SymInt)
DEFINE_BINARY(max, std::max, sym_max, SymInt)
// clang-format on

SymInt::operator SymFloat() const {
  if (auto ma = maybe_as_int()) {
    return SymFloat(double(*ma));
  } else {
    return SymFloat(toSymNodeImplUnowned()->sym_float());
  }
}

SymNode SymInt::wrap_node(const SymNode& base) const {
  if (auto ma = maybe_as_int()) {
    return base->wrap_int(*ma);
  } else {
    return toSymNode();
  }
}

SymInt SymInt::clone() const {
  if (auto ma = maybe_as_int()) {
    return SymInt(*ma);
  } else {
    return SymInt(toSymNodeImplUnowned()->clone());
  }
}

int64_t SymInt::guard_int(const char* file, int64_t line) const {
  if (auto ma = maybe_as_int()) {
    return *ma;
  } else {
    return toSymNodeImplUnowned()->guard_int(file, line);
  }
}

SymInt operator-(const SymInt& s) {
  if (auto ma = s.maybe_as_int()) {
    return SymInt(-*ma);
  } else {
    return SymInt(s.toSymNodeImplUnowned()->neg());
  }
}

void SymInt::operator*=(const SymInt& sci) {
  *this = *this * sci;
}

void SymInt::operator/=(const SymInt& sci) {
  *this = *this / sci;
}

void SymInt::operator+=(const SymInt& sci) {
  *this = *this + sci;
}

std::ostream& operator<<(std::ostream& os, const SymInt& s) {
  if (s.is_heap_allocated()) {
    os << s.toSymNodeImplUnowned()->str();
  } else {
    os << s.as_int_unchecked();
  }
  return os;
}

// This template lets us not do a refcount bump when we do an
// identity conversion
template <typename T>
struct Convert {};

template <>
struct Convert<SymInt> {
  const SymInt& operator()(const SymInt& a) {
    return a;
  }
};

template <>
struct Convert<SymFloat> {
  SymFloat operator()(const SymInt& a) {
    return a;
  }
};

#define DEFINE_SYMINT_OP_INTONLY(scalar_t, RetTy) \
  RetTy operator%(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) % RetTy(b);        \
  };                                              \
  RetTy operator%(scalar_t a, const SymInt& b) {  \
    return RetTy(a) % Convert<RetTy>()(b);        \
  };

#define DEFINE_SYMINT_OP(scalar_t, RetTy)        \
  RetTy operator+(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) + RetTy(b);       \
  };                                             \
  RetTy operator-(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) - RetTy(b);       \
  };                                             \
  RetTy operator*(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) * RetTy(b);       \
  };                                             \
  RetTy operator/(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) / RetTy(b);       \
  };                                             \
  RetTy operator+(scalar_t a, const SymInt& b) { \
    return RetTy(a) + Convert<RetTy>()(b);       \
  };                                             \
  RetTy operator-(scalar_t a, const SymInt& b) { \
    return RetTy(a) - Convert<RetTy>()(b);       \
  };                                             \
  RetTy operator*(scalar_t a, const SymInt& b) { \
    return RetTy(a) * Convert<RetTy>()(b);       \
  };                                             \
  RetTy operator/(scalar_t a, const SymInt& b) { \
    return RetTy(a) / Convert<RetTy>()(b);       \
  };                                             \
  bool operator==(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) == RetTy(b);      \
  };                                             \
  bool operator!=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) != RetTy(b);      \
  };                                             \
  bool operator<(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) < RetTy(b);       \
  };                                             \
  bool operator<=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) <= RetTy(b);      \
  };                                             \
  bool operator>(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) > RetTy(b);       \
  };                                             \
  bool operator>=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) >= RetTy(b);      \
  };                                             \
  bool operator==(scalar_t a, const SymInt& b) { \
    return RetTy(a) == Convert<RetTy>()(b);      \
  };                                             \
  bool operator!=(scalar_t a, const SymInt& b) { \
    return RetTy(a) != Convert<RetTy>()(b);      \
  };                                             \
  bool operator<(scalar_t a, const SymInt& b) {  \
    return RetTy(a) < Convert<RetTy>()(b);       \
  };                                             \
  bool operator<=(scalar_t a, const SymInt& b) { \
    return RetTy(a) <= Convert<RetTy>()(b);      \
  };                                             \
  bool operator>(scalar_t a, const SymInt& b) {  \
    return RetTy(a) > Convert<RetTy>()(b);       \
  };                                             \
  bool operator>=(scalar_t a, const SymInt& b) { \
    return RetTy(a) >= Convert<RetTy>()(b);      \
  };

DEFINE_SYMINT_OP_INTONLY(int64_t, SymInt)
DEFINE_SYMINT_OP_INTONLY(int32_t, SymInt)
DEFINE_SYMINT_OP_INTONLY(uint64_t, SymInt)
DEFINE_SYMINT_OP_INTONLY(uint32_t, SymInt)
DEFINE_SYMINT_OP(int64_t, SymInt)
DEFINE_SYMINT_OP(int32_t, SymInt) // make sure constants work
DEFINE_SYMINT_OP(uint64_t, SymInt)
DEFINE_SYMINT_OP(uint32_t, SymInt)
DEFINE_SYMINT_OP(double, SymFloat)
DEFINE_SYMINT_OP(float, SymFloat) // just for completeness

#if defined(__APPLE__)
DEFINE_SYMINT_OP_INTONLY(size_t, SymInt) // needed for osx
DEFINE_SYMINT_OP(size_t, SymInt) // needed for osx
#endif

} // namespace c10
