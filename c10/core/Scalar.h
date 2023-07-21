#pragma once

#include <stdint.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <c10/core/OptionalRef.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/TypeCast.h>
#include <c10/util/intrusive_ptr.h>

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

  AT_FORALL_SCALAR_TYPES_AND5(
      Half,
      BFloat16,
      Float8_e5m2,
      Float8_e4m3fn,
      ComplexHalf,
      DEFINE_IMPLICIT_CTOR)
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
    v.i = convert<int64_t, bool>(vv);
  }

  template <
      typename T,
      typename std::enable_if<std::is_same<T, c10::SymBool>::value, bool>::
          type* = nullptr>
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
    } else if (Tag::HAS_si == tag) {                                  \
      TORCH_CHECK(false, "tried to get " #name " out of SymInt")      \
    } else if (Tag::HAS_sd == tag) {                                  \
      TORCH_CHECK(false, "tried to get " #name " out of SymFloat")    \
    } else if (Tag::HAS_sb == tag) {                                  \
      TORCH_CHECK(false, "tried to get " #name " out of SymBool")     \
    }                                                                 \
    TORCH_CHECK(false)                                                \
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)

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
      typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
  bool equal(T num) const {
    if (isComplex()) {
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      auto val = v.z;
      return (val.real() == num) && (val.imag() == T());
    } else if (isFloatingPoint()) {
      TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
      return v.d == num;
    } else if (isIntegral(/*includeBool=*/false)) {
      TORCH_CHECK(!isSymbolic(), "NYI SymInt equality");
      return v.i == num;
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
      typename std::enable_if<c10::is_complex<T>::value, int>::type = 0>
  bool equal(T num) const {
    if (isComplex()) {
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return v.z == num;
    } else if (isFloatingPoint()) {
      TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
      return (v.d == num.real()) && (num.imag() == T());
    } else if (isIntegral(/*includeBool=*/false)) {
      TORCH_CHECK(!isSymbolic(), "NYI SymInt equality");
      return (v.i == num.real()) && (num.imag() == T());
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
    if (sb.is_symbolic()) {
      tag = Tag::HAS_sb;
      v.p = std::move(sb).release();
    } else {
      tag = Tag::HAS_b;
      v.d = sb.as_bool_unchecked();
    }
  }

  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC
 private:
  enum class Tag { HAS_d, HAS_i, HAS_z, HAS_b, HAS_sd, HAS_si, HAS_sb };

  // NB: assumes that self has already been cleared
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
    c10::complex<double> z;
    c10::intrusive_ptr_target* p;
    v_t() {} // default constructor
  } v;

  template <
      typename T,
      typename std::enable_if<
          std::is_integral<T>::value && !std::is_same<T, bool>::value,
          bool>::type* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.i = convert<decltype(v.i), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if<
          !std::is_integral<T>::value && !c10::is_complex<T>::value,
          bool>::type* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.d = convert<decltype(v.d), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if<c10::is_complex<T>::value, bool>::type* = nullptr>
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
#undef DEFINE_TO

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()
