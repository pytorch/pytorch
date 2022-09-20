#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <c10/core/OptionalRef.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/TypeCast.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

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

#define DEFINE_IMPLICIT_CTOR(type, name) \
  Scalar(type vv) : Scalar(vv, true) {}

  void destroy() {
    if (Tag::HAS_si == tag) {
      v.si.release_();
    }
  }

  ~Scalar() {
    destroy();
  }

  AT_FORALL_SCALAR_TYPES_AND3(Half, BFloat16, ComplexHalf, DEFINE_IMPLICIT_CTOR)
  AT_FORALL_COMPLEX_TYPES(DEFINE_IMPLICIT_CTOR)

#undef DEFINE_IMPLICIT_CTOR

  // Value* is both implicitly convertible to SymbolicVariable and bool which
  // causes ambiguity error. Specialized constructor for bool resolves this
  // problem.
  template <
      typename T,
      typename std::enable_if<std::is_same<T, bool>::value, bool>::type* =
          nullptr>
  Scalar(T vv) : tag(Tag::HAS_b) {
    v.u.i = convert<int64_t, bool>(vv);
  }

#define DEFINE_ACCESSOR(type, name)                                     \
  type to##name() const {                                               \
    if (Tag::HAS_d == tag) {                                            \
      return checked_convert<type, double>(v.u.d, #type);               \
    } else if (Tag::HAS_z == tag) {                                     \
      return checked_convert<type, c10::complex<double>>(v.u.z, #type); \
    }                                                                   \
    if (Tag::HAS_b == tag) {                                            \
      return checked_convert<type, bool>(v.u.i, #type);                 \
    } else if (Tag::HAS_i == tag) {                                     \
      return checked_convert<type, int64_t>(v.u.i, #type);              \
    } else if (Tag::HAS_si == tag) {                                    \
      TORCH_CHECK(false)                                                \
    }                                                                   \
    TORCH_CHECK(false)                                                  \
  }

  SymInt toSymInt() const {
    if (Tag::HAS_si == tag) {
      return v.si;
    } else {
      return toLong();
    }
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)

  // also support scalar.to<int64_t>();
  // Deleted for unsupported types, but specialized below for supported types
  template <typename T>
  T to() const = delete;

  // audit uses of data_ptr
  const void* data_ptr() const {
    // error on symint
    return static_cast<const void*>(&v);
  }

#undef DEFINE_ACCESSOR
  bool isFloatingPoint() const {
    return Tag::HAS_d == tag;
  }

  C10_DEPRECATED_MESSAGE(
      "isIntegral is deprecated. Please use the overload with 'includeBool' parameter instead.")
  bool isIntegral() const {
    return Tag::HAS_i == tag || Tag::HAS_si == tag;
  }
  bool isIntegral(bool includeBool) const {
    return Tag::HAS_i == tag || Tag::HAS_si == tag ||
        (includeBool && isBoolean());
  }

  bool isComplex() const {
    return Tag::HAS_z == tag;
  }
  bool isBoolean() const {
    return Tag::HAS_b == tag;
  }
  // nb: does not include normal ints
  bool isSymInt() const {
    return Tag::HAS_si == tag;
  }

  C10_ALWAYS_INLINE Scalar& operator=(Scalar&& other) {
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
      typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
  bool equal(T num) const {
    if (isComplex()) {
      auto val = v.u.z;
      return (val.real() == num) && (val.imag() == T());
    } else if (isFloatingPoint()) {
      return v.u.d == num;
    } else if (isIntegral(/*includeBool=*/false)) {
      // test if symint or not
      // guard on bool
      return v.u.i == num;
    } else {
      // boolean scalar does not equal to a non boolean value
      return false;
    }
  }

  template <
      typename T,
      typename std::enable_if<c10::is_complex<T>::value, int>::type = 0>
  bool equal(T num) const {
    if (isComplex()) {
      return v.u.z == num;
    } else if (isFloatingPoint()) {
      return (v.u.d == num.real()) && (num.imag() == T());
    } else if (isIntegral(/*includeBool=*/false)) {
      // test symint here
      return (v.u.i == num.real()) && (num.imag() == T());
    } else {
      // boolean scalar does not equal to a non boolean value
      return false;
    }
  }

  bool equal(bool num) const {
    if (isBoolean()) {
      return static_cast<bool>(v.u.i) == num;
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

  Scalar(const Scalar& rhs) : Scalar(rhs.v, rhs.tag) {}

  Scalar(c10::SymInt si) : tag(Tag::HAS_si) {
    // test if int and turn to int
    v.si = si;
  }

  C10_ALWAYS_INLINE void moveFrom(Scalar&& rhs) noexcept {
    if (rhs.tag == Tag::HAS_si) {
      new (&v.si) c10::SymInt(std::move(rhs.v.si));
      rhs.v.si.release_();
    } else {
      v.u = rhs.v.u;
    }
    tag = rhs.tag;
    rhs.clearToInt();
  }

  void clearToInt() noexcept {
    v.u.i = 0;
    tag = Tag::HAS_i;
  }

  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC
 private:
  enum class Tag { HAS_d, HAS_i, HAS_z, HAS_b, HAS_si, HAS_sf };

  union Payload {
    // See [TriviallyCopyablePayload] in IValue.h,
    union TriviallyCopyablePayload {
      TriviallyCopyablePayload() : i(0) {}
      double d;
      int64_t i;
      c10::complex<double> z;
    } u;
    Payload() : u() {}
    ~Payload() {}
    c10::SymInt si;
  } v;

  Tag tag;
  Payload payload;

  Scalar(const Payload& p, Scalar::Tag t) : tag(t) {
    if (t == Tag::HAS_si) {
      v.si = p.si;
    } else {
      v.u = p.u;
    }
  }

  template <
      typename T,
      typename std::enable_if<
          std::is_integral<T>::value && !std::is_same<T, bool>::value,
          bool>::type* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.u.i = convert<decltype(v.u.i), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if<
          !std::is_integral<T>::value && !c10::is_complex<T>::value,
          bool>::type* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.u.d = convert<decltype(v.u.d), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if<c10::is_complex<T>::value, bool>::type* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_z) {
    v.u.z = convert<decltype(v.u.z), T>(vv);
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
#undef DEFINE_TO

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()
