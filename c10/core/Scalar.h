#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <utility>

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Half.h>

namespace c10 {

/**
 * Scalar represents a 0-dimensional tensor which contains a single element.
 * Unlike a tensor, numeric literals (in C++) are implicitly convertible to
 * Scalar (which is why, for example, we provide both add(Tensor) and
 * add(Scalar) overloads for many operations). It may also be used in
 * circumstances where you statically know a tensor is 0-dim and single size,
 * but don't know it's type.
 */
class C10_API Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CTOR(type, name, member)      \
  Scalar(type vv) : tag(Tag::HAS_##member) {          \
    v.member = convert<decltype(v.member), type>(vv); \
  }
  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC

  AT_FORALL_SCALAR_TYPES_EXCEPT_QINT(DEFINE_IMPLICIT_CTOR)

#undef DEFINE_IMPLICIT_CTOR

  // Value* is both implicitly convertible to SymbolicVariable and bool which
  // causes ambiguosity error. Specialized constructor for bool resolves this
  // problem.
  template <
      typename T,
      typename std::enable_if<std::is_same<T, bool>::value, bool>::type* =
          nullptr>
  Scalar(T vv) : tag(Tag::HAS_i) {
    v.i = convert<decltype(v.i), bool>(vv);
  }

#define DEFINE_IMPLICIT_COMPLEX_CTOR(type, name, member) \
  Scalar(type vv) : tag(Tag::HAS_##member) {             \
    v.member[0] = c10::convert<double>(vv.real());       \
    v.member[1] = c10::convert<double>(vv.imag());       \
  }

  DEFINE_IMPLICIT_COMPLEX_CTOR(at::ComplexHalf, ComplexHalf, z)
  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<float>, ComplexFloat, z)
  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<double>, ComplexDouble, z)

#undef DEFINE_IMPLICIT_COMPLEX_CTOR

#define DEFINE_ACCESSOR(type, name, member)               \
  type to##name() const {                                 \
    if (Tag::HAS_d == tag) {                              \
      return checked_convert<type, double>(v.d, #type);   \
    } else if (Tag::HAS_z == tag) {                       \
      return checked_convert<type, std::complex<double>>( \
          {v.z[0], v.z[1]}, #type);                       \
    } else {                                              \
      return checked_convert<type, int64_t>(v.i, #type);  \
    }                                                     \
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_AND_QINT(
      DEFINE_ACCESSOR)

  // also support scalar.to<int64_t>();
  template <typename T>
  T to();

#undef DEFINE_ACCESSOR
  bool isFloatingPoint() const {
    return Tag::HAS_d == tag;
  }
  bool isIntegral() const {
    return Tag::HAS_i == tag;
  }
  bool isComplex() const {
    return Tag::HAS_z == tag;
  }

  Scalar operator-() const;

 private:
  enum class Tag { HAS_d, HAS_i, HAS_z };
  Tag tag;
  union {
    double d;
    int64_t i;
    // Can't do put std::complex in the union, because it triggers
    // an nvcc bug:
    //    error: designator may not specify a non-POD subobject
    double z[2];
  } v;
};

// define the scalar.to<int64_t>() specializations
template <typename T>
inline T Scalar::to() {
  throw std::runtime_error("to() cast to unexpected type.");
}

#define DEFINE_TO(T, name, _) \
  template <>                 \
  inline T Scalar::to<T>() {  \
    return to##name();        \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_AND_QINT(DEFINE_TO)
#undef DEFINE_TO
} // namespace c10
