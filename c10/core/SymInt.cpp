#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/safe_numerics.h>
#include <functional>

namespace c10 {

// Precondition: data_ has a large negative number that should be
// treated as a constant.  It is NOT a valid pointer.  In other words,
// SymInt has temporarily violated invariants
// Postcondition: invariants on SymInt are fixed
void SymInt::promote_to_negative() {
  auto s =
      SymInt(SymNode(c10::make_intrusive<ConstantSymNodeImpl<int64_t>>(data_)));
  // Similar to move operator=, but do NOT release data_
  data_ = s.data_;
  s.data_ = 0;
}

std::optional<int64_t> SymInt::maybe_as_int_slow_path() const {
  auto* node = toSymNodeImplUnowned();
  if (auto c = node->constant_int()) {
    return c;
  }
  return node->maybe_as_int();
}

SymNode SymInt::toSymNode() const {
  TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
      is_heap_allocated(), "SymInt::toSymNode is_heap_allocated");
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymInt::SymInt(SymNode sin_sp) {
  TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
      sin_sp->is_int(), "SymInt::SymInt sin_sp->is_int()");
  auto ptr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sin_sp.release()));
  auto rep = (ptr & ~MASK) | IS_SYM;
  data_ = static_cast<int64_t>(rep);
}

bool SymInt::has_hint() const {
  if (!is_heap_allocated()) {
    return true;
  }
  return toSymNodeImplUnowned()->has_hint();
}

#define DEFINE_BINARY(API, METHOD, RET)                              \
  RET SymInt::API(const SymInt& sci) const {                         \
    if (auto ma = maybe_as_int()) {                                  \
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(                              \
          !sci.maybe_as_int(),                                       \
          "should have hit fast path in the header in this case.");  \
      auto b = sci.toSymNode();                                      \
      return RET(b->wrap_int(*ma)->METHOD(b));                       \
    } else {                                                         \
      if (auto mb = sci.maybe_as_int()) {                            \
        auto a = toSymNodeImplUnowned();                             \
        return RET(a->METHOD(a->wrap_int(*mb)));                     \
      } else {                                                       \
        return RET(toSymNodeImplUnowned()->METHOD(sci.toSymNode())); \
      }                                                              \
    }                                                                \
  }

DEFINE_BINARY(operator_add_slow_path, add, SymInt)
DEFINE_BINARY(operator_sub_slow_path, sub, SymInt)
DEFINE_BINARY(operator_mul_slow_path, mul, SymInt)
DEFINE_BINARY(operator_div_slow_path, floordiv, SymInt)
DEFINE_BINARY(operator_mod_slow_path, mod, SymInt)
DEFINE_BINARY(sym_eq_slow_path, eq, SymBool)
DEFINE_BINARY(sym_ne_slow_path, ne, SymBool)
DEFINE_BINARY(sym_lt_slow_path, lt, SymBool)
DEFINE_BINARY(sym_le_slow_path, le, SymBool)
DEFINE_BINARY(sym_gt_slow_path, gt, SymBool)
DEFINE_BINARY(sym_ge_slow_path, ge, SymBool)
DEFINE_BINARY(min_slow_path, sym_min, SymInt)
DEFINE_BINARY(max_slow_path, sym_max, SymInt)

SymInt::operator SymFloat() const {
  if (auto ma = maybe_as_int()) {
    return SymFloat(double(*ma));
  } else {
    return SymFloat(toSymNodeImplUnowned()->sym_float());
  }
}

bool SymInt::is_same(const SymInt& other) const {
  if (is_heap_allocated() != other.is_heap_allocated()) {
    return false;
  }
  // Both not heap allocated
  if (!is_heap_allocated() && this->operator!=(other)) {
    return false;
  }
  // Both heap allocated
  if (is_heap_allocated() &&
      toSymNodeImplUnowned() != other.toSymNodeImplUnowned()) {
    return false;
  }
  return true;
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

bool SymInt::expect_size(const char* file, int64_t line) const {
  if (auto ma = maybe_as_int()) {
    return *ma >= 0;
  } else {
    return toSymNodeImplUnowned()->expect_size(file, line);
  }
}

SymInt operator-(const SymInt& s) {
  if (auto ma = s.maybe_as_int()) {
    const auto val = *ma;
    // Note: Result of `-std::numeric_limits<decltype(val)>::min()` is undefined
    // But on many platforms it equals to self + setting Carry/Overflow flags
    // Which in optimized code affects results of `check_range` condition
    // Workaround by using ternary that avoids alterning the flags
#if C10_HAS_BUILTIN_OVERFLOW()
    std::decay_t<decltype(val)> out = 0;
    if (C10_UNLIKELY(__builtin_sub_overflow(out, val, &out))) {
      return SymInt(val);
    }
    return SymInt(out);
#else
    constexpr auto val_min = std::numeric_limits<decltype(val)>::min();
    return SymInt(val != val_min ? -val : val_min);
#endif
  } else {
    return SymInt(s.toSymNodeImplUnowned()->neg());
  }
}

void SymInt::operator_imul_slow_path(const SymInt& sci) {
  *this = *this * sci;
}

void SymInt::operator_idiv_slow_path(const SymInt& sci) {
  *this = *this / sci;
}

void SymInt::operator_iadd_slow_path(const SymInt& sci) {
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
  }                                               \
  RetTy operator%(scalar_t a, const SymInt& b) {  \
    return RetTy(a) % Convert<RetTy>()(b);        \
  }

#define DEFINE_SYMINT_OP(scalar_t, RetTy)        \
  RetTy operator+(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) + RetTy(b);       \
  }                                              \
  RetTy operator-(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) - RetTy(b);       \
  }                                              \
  RetTy operator*(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) * RetTy(b);       \
  }                                              \
  RetTy operator/(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) / RetTy(b);       \
  }                                              \
  RetTy operator+(scalar_t a, const SymInt& b) { \
    return RetTy(a) + Convert<RetTy>()(b);       \
  }                                              \
  RetTy operator-(scalar_t a, const SymInt& b) { \
    return RetTy(a) - Convert<RetTy>()(b);       \
  }                                              \
  RetTy operator*(scalar_t a, const SymInt& b) { \
    return RetTy(a) * Convert<RetTy>()(b);       \
  }                                              \
  RetTy operator/(scalar_t a, const SymInt& b) { \
    return RetTy(a) / Convert<RetTy>()(b);       \
  }                                              \
  bool operator==(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) == RetTy(b);      \
  }                                              \
  bool operator!=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) != RetTy(b);      \
  }                                              \
  bool operator<(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) < RetTy(b);       \
  }                                              \
  bool operator<=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) <= RetTy(b);      \
  }                                              \
  bool operator>(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) > RetTy(b);       \
  }                                              \
  bool operator>=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) >= RetTy(b);      \
  }                                              \
  bool operator==(scalar_t a, const SymInt& b) { \
    return RetTy(a) == Convert<RetTy>()(b);      \
  }                                              \
  bool operator!=(scalar_t a, const SymInt& b) { \
    return RetTy(a) != Convert<RetTy>()(b);      \
  }                                              \
  bool operator<(scalar_t a, const SymInt& b) {  \
    return RetTy(a) < Convert<RetTy>()(b);       \
  }                                              \
  bool operator<=(scalar_t a, const SymInt& b) { \
    return RetTy(a) <= Convert<RetTy>()(b);      \
  }                                              \
  bool operator>(scalar_t a, const SymInt& b) {  \
    return RetTy(a) > Convert<RetTy>()(b);       \
  }                                              \
  bool operator>=(scalar_t a, const SymInt& b) { \
    return RetTy(a) >= Convert<RetTy>()(b);      \
  }

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
