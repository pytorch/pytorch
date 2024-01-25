#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <c10/core/OptionalRef.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/TypeCast.h>
#include <c10/util/complex.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {

/**
 * Scalar represents a 0-dimensional tensor which contains a single element.
 * Unlike a tensor, numeric literals (in C++) are implicitly convertible to
 * Scalar (which is why, for example, we provide both add(Tensor) and
 * add(Scalar) overloads for many operations). It may also be used in
 * circumstances where you statically know a tensor is 0-dim and single size,
 * but don't know its type.
 */
class C10_API Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

  void destroy() {
    if (Tag::HAS_si == tag || Tag::HAS_sd == tag || Tag::HAS_sb == tag) {
      raw::intrusive_ptr::decref(v.p);
      v.p = nullptr;
    }
  }

  ~Scalar() {
    destroy();
  }

#define DEFINE_IMPLICIT_CTOR(type, name) \
  Scalar(type vv) : Scalar(vv, true) {}

  AT_FORALL_SCALAR_TYPES_AND7(
      Half,
      BFloat16,
      Float8_e5m2,
      Float8_e4m3fn,
      Float8_e5m2fnuz,
      Float8_e4m3fnuz,
      ComplexHalf,
      DEFINE_IMPLICIT_CTOR)
  AT_FORALL_COMPLEX_TYPES(DEFINE_IMPLICIT_CTOR)

  // Helper constructors to allow Scalar creation from long and long long types
  // As std::is_same_v<long, long long> is false(except Android), one needs to
  // provide a constructor from either long or long long in addition to one from
  // int64_t
#if defined(__APPLE__) || defined(__MACOSX)
  static_assert(
      std::is_same_v<long long, int64_t>,
      "int64_t is the same as long long on MacOS");
  Scalar(long vv) : Scalar(vv, true) {}
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  static_assert(
      std::is_same_v<long, int64_t>,
      "int64_t is the same as long on Linux");
  Scalar(long long vv) : Scalar(vv, true) {}
#endif

  Scalar(uint16_t vv) : Scalar(vv, true) {}
  Scalar(uint32_t vv) : Scalar(vv, true) {}
  Scalar(uint64_t vv) {
    if (vv > static_cast<uint64_t>(INT64_MAX)) {
      tag = Tag::HAS_u;
      v.u = vv;
    } else {
      tag = Tag::HAS_i;
      // NB: no need to use convert, we've already tested convertibility
      v.i = static_cast<int64_t>(vv);
    }
  }

#undef DEFINE_IMPLICIT_CTOR

  // Value* is both implicitly convertible to SymbolicVariable and bool which
  // causes ambiguity error. Specialized constructor for bool resolves this
  // problem.
  template <
      typename T,
      typename std::enable_if_t<std::is_same_v<T, bool>, bool>* = nullptr>
  Scalar(T vv) : tag(Tag::HAS_b) {
    v.i = convert<int64_t, bool>(vv);
  }

  template <
      typename T,
      typename std::enable_if_t<std::is_same_v<T, c10::SymBool>, bool>* =
          nullptr>
  Scalar(T vv) : tag(Tag::HAS_sb) {
    v.i = convert<int64_t, c10::SymBool>(vv);
  }

#define DEFINE_ACCESSOR(type, name)                                   \
  type to##name() const {                                             \
    if (Tag::HAS_d == tag) {                                          \
      return checked_convert<type, double>(v.d, #type);               \
    } else if (Tag::HAS_z == tag) {                                   \
      return checked_convert<type, c10::complex<double>>(v.z, #type); \
    }                                                                 \
    if (Tag::HAS_b == tag) {                                          \
      return checked_convert<type, bool>(v.i, #type);                 \
    } else if (Tag::HAS_i == tag) {                                   \
      return checked_convert<type, int64_t>(v.i, #type);              \
    } else if (Tag::HAS_u == tag) {                                   \
      return checked_convert<type, uint64_t>(v.u, #type);             \
    } else if (Tag::HAS_si == tag) {                                  \
      return checked_convert<type, int64_t>(                          \
          toSymInt().guard_int(__FILE__, __LINE__), #type);           \
    } else if (Tag::HAS_sd == tag) {                                  \
      return checked_convert<type, int64_t>(                          \
          toSymFloat().guard_float(__FILE__, __LINE__), #type);       \
    } else if (Tag::HAS_sb == tag) {                                  \
      return checked_convert<type, int64_t>(                          \
          toSymBool().guard_bool(__FILE__, __LINE__), #type);         \
    }                                                                 \
    TORCH_CHECK(false)                                                \
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)
  DEFINE_ACCESSOR(uint16_t, UInt16)
  DEFINE_ACCESSOR(uint32_t, UInt32)
  DEFINE_ACCESSOR(uint64_t, UInt64)

#undef DEFINE_ACCESSOR

  SymInt toSymInt() const {
    if (Tag::HAS_si == tag) {
      return c10::SymInt(intrusive_ptr<SymNodeImpl>::reclaim_copy(
          static_cast<SymNodeImpl*>(v.p)));
    } else {
      return toLong();
    }
  }

  SymFloat toSymFloat() const {
    if (Tag::HAS_sd == tag) {
      return c10::SymFloat(intrusive_ptr<SymNodeImpl>::reclaim_copy(
          static_cast<SymNodeImpl*>(v.p)));
    } else {
      return toDouble();
    }
  }

  SymBool toSymBool() const {
    if (Tag::HAS_sb == tag) {
      return c10::SymBool(intrusive_ptr<SymNodeImpl>::reclaim_copy(
          static_cast<SymNodeImpl*>(v.p)));
    } else {
      return toBool();
    }
  }

  // also support scalar.to<int64_t>();
  // Deleted for unsupported types, but specialized below for supported types
  template <typename T>
  T to() const = delete;

  // audit uses of data_ptr
  const void* data_ptr() const {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    return static_cast<const void*>(&v);
  }

  bool isFloatingPoint() const {
    return Tag::HAS_d == tag || Tag::HAS_sd == tag;
  }

  C10_DEPRECATED_MESSAGE(
      "isIntegral is deprecated. Please use the overload with 'includeBool' parameter instead.")
  bool isIntegral() const {
    return Tag::HAS_i == tag || Tag::HAS_si == tag || Tag::HAS_u == tag;
  }
  bool isIntegral(bool includeBool) const {
    return Tag::HAS_i == tag || Tag::HAS_si == tag || Tag::HAS_u == tag ||
        (includeBool && isBoolean());
  }

  bool isComplex() const {
    return Tag::HAS_z == tag;
  }
  bool isBoolean() const {
    return Tag::HAS_b == tag || Tag::HAS_sb == tag;
  }

  // you probably don't actually want these; they're mostly for testing
  bool isSymInt() const {
    return Tag::HAS_si == tag;
  }
  bool isSymFloat() const {
    return Tag::HAS_sd == tag;
  }
  bool isSymBool() const {
    return Tag::HAS_sb == tag;
  }

  bool isSymbolic() const {
    return Tag::HAS_si == tag || Tag::HAS_sd == tag || Tag::HAS_sb == tag;
  }

  C10_ALWAYS_INLINE Scalar& operator=(Scalar&& other) noexcept {
    if (&other == this) {
      return *this;
    }

    destroy();
    moveFrom(std::move(other));
    return *this;
  }

  C10_ALWAYS_INLINE Scalar& operator=(const Scalar& other) {
    if (&other == this) {
      return *this;
    }

    *this = Scalar(other);
    return *this;
  }

  Scalar operator-() const;
  Scalar conj() const;
  Scalar log() const;

  template <
      typename T,
      typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
  bool equal(T num) const {
    if (isComplex()) {
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      auto val = v.z;
      return (val.real() == num) && (val.imag() == T());
    } else if (isFloatingPoint()) {
      TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
      return v.d == num;
    } else if (tag == Tag::HAS_i) {
      if (overflows<T>(v.i, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.i) == num;
      }
    } else if (tag == Tag::HAS_u) {
      if (overflows<T>(v.u, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.u) == num;
      }
    } else if (tag == Tag::HAS_si) {
      TORCH_INTERNAL_ASSERT(false, "NYI SymInt equality");
    } else if (isBoolean()) {
      // boolean scalar does not equal to a non boolean value
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return false;
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  template <
      typename T,
      typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
  bool equal(T num) const {
    if (isComplex()) {
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return v.z == num;
    } else if (isFloatingPoint()) {
      TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
      return (v.d == num.real()) && (num.imag() == T());
    } else if (tag == Tag::HAS_i) {
      if (overflows<T>(v.i, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.i) == num.real() && num.imag() == T();
      }
    } else if (tag == Tag::HAS_u) {
      if (overflows<T>(v.u, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.u) == num.real() && num.imag() == T();
      }
    } else if (tag == Tag::HAS_si) {
      TORCH_INTERNAL_ASSERT(false, "NYI SymInt equality");
    } else if (isBoolean()) {
      // boolean scalar does not equal to a non boolean value
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return false;
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  bool equal(bool num) const {
    if (isBoolean()) {
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return static_cast<bool>(v.i) == num;
    } else {
      return false;
    }
  }

  ScalarType type() const {
    if (isComplex()) {
      return ScalarType::ComplexDouble;
    } else if (isFloatingPoint()) {
      return ScalarType::Double;
    } else if (isIntegral(/*includeBool=*/false)) {
      // Represent all integers as long, UNLESS it is unsigned and therefore
      // unrepresentable as long
      if (Tag::HAS_u == tag) {
        return ScalarType::UInt64;
      }
      return ScalarType::Long;
    } else if (isBoolean()) {
      return ScalarType::Bool;
    } else {
      throw std::runtime_error("Unknown scalar type.");
    }
  }

  Scalar(Scalar&& rhs) noexcept : tag(rhs.tag) {
    moveFrom(std::move(rhs));
  }

  Scalar(const Scalar& rhs) : tag(rhs.tag), v(rhs.v) {
    if (isSymbolic()) {
      c10::raw::intrusive_ptr::incref(v.p);
    }
  }

  Scalar(c10::SymInt si) {
    if (auto m = si.maybe_as_int()) {
      tag = Tag::HAS_i;
      v.i = *m;
    } else {
      tag = Tag::HAS_si;
      v.p = std::move(si).release();
    }
  }

  Scalar(c10::SymFloat sd) {
    if (sd.is_symbolic()) {
      tag = Tag::HAS_sd;
      v.p = std::move(sd).release();
    } else {
      tag = Tag::HAS_d;
      v.d = sd.as_float_unchecked();
    }
  }

  Scalar(c10::SymBool sb) {
    if (auto m = sb.maybe_as_bool()) {
      tag = Tag::HAS_b;
      v.i = *m;
    } else {
      tag = Tag::HAS_sb;
      v.p = std::move(sb).release();
    }
  }

  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC
 private:
  enum class Tag { HAS_d, HAS_i, HAS_u, HAS_z, HAS_b, HAS_sd, HAS_si, HAS_sb };

  // Note [Meaning of HAS_u]
  // ~~~~~~~~~~~~~~~~~~~~~~~
  // HAS_u is a bit special.  On its face, it just means that we
  // are holding an unsigned integer.  However, we generally don't
  // distinguish between different bit sizes in Scalar (e.g., we represent
  // float as double), instead, it represents a mathematical notion
  // of some quantity (integral versus floating point).  So actually,
  // HAS_u is used solely to represent unsigned integers that could
  // not be represented as a signed integer.  That means only uint64_t
  // potentially can get this tag; smaller types like uint8_t fits into a
  // regular int and so for BC reasons we keep as an int.

  // NB: assumes that self has already been cleared
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  C10_ALWAYS_INLINE void moveFrom(Scalar&& rhs) noexcept {
    v = rhs.v;
    tag = rhs.tag;
    if (rhs.tag == Tag::HAS_si || rhs.tag == Tag::HAS_sd ||
        rhs.tag == Tag::HAS_sb) {
      // Move out of scalar
      rhs.tag = Tag::HAS_i;
      rhs.v.i = 0;
    }
  }

  Tag tag;

  union v_t {
    double d{};
    int64_t i;
    // See Note [Meaning of HAS_u]
    uint64_t u;
    c10::complex<double> z;
    c10::intrusive_ptr_target* p;
    // NOLINTNEXTLINE(modernize-use-equals-default)
    v_t() {} // default constructor
  } v;

  template <
      typename T,
      typename std::enable_if_t<
          std::is_integral_v<T> && !std::is_same_v<T, bool>,
          bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.i = convert<decltype(v.i), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if_t<
          !std::is_integral_v<T> && !c10::is_complex<T>::value,
          bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.d = convert<decltype(v.d), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if_t<c10::is_complex<T>::value, bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_z) {
    v.z = convert<decltype(v.z), T>(vv);
  }
};

using OptionalScalarRef = c10::OptionalRef<Scalar>;

// define the scalar.to<int64_t>() specializations
#define DEFINE_TO(T, name)         \
  template <>                      \
  inline T Scalar::to<T>() const { \
    return to##name();             \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO)
DEFINE_TO(uint16_t, UInt16)
DEFINE_TO(uint32_t, UInt32)
DEFINE_TO(uint64_t, UInt64)
#undef DEFINE_TO

} // namespace c10
