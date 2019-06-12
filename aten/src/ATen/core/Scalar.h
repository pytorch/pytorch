#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "ATen/core/ATenGeneral.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/Half.h"

namespace at {

class Tensor;

class CAFFE2_API Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CTOR(type,name,member) \
  Scalar(type vv) \
  : tag(Tag::HAS_##member) { \
    v . member = convert<decltype(v.member),type>(vv); \
  }
  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC

  AT_FORALL_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR)

#undef DEFINE_IMPLICIT_CTOR

#define DEFINE_IMPLICIT_COMPLEX_CTOR(type, name, member) \
  Scalar(type vv) : tag(Tag::HAS_##member) {             \
    v.member[0] = c10::convert<double>(vv.real());       \
    v.member[1] = c10::convert<double>(vv.imag());       \
  }

  DEFINE_IMPLICIT_COMPLEX_CTOR(at::ComplexHalf,ComplexHalf,z)
  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<float>,ComplexFloat,z)
  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<double>,ComplexDouble,z)

#undef DEFINE_IMPLICIT_COMPLEX_CTOR

#define DEFINE_ACCESSOR(type,name,member) \
  type to##name () const { \
    if (Tag::HAS_d == tag) { \
      return checked_convert<type, double>(v.d, #type); \
    } else if (Tag::HAS_z == tag) { \
      return checked_convert<type, std::complex<double>>({v.z[0], v.z[1]}, #type); \
    } else { \
      return checked_convert<type, int64_t>(v.i, #type); \
    } \
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ACCESSOR)

  //also support scalar.to<int64_t>();
  template<typename T>
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
  friend struct Type;
};

// define the scalar.to<int64_t>() specializations
template<typename T>
inline T Scalar::to() {
  throw std::runtime_error("to() cast to unexpected type.");
}

#define DEFINE_TO(T,name,_) \
template<> \
inline T Scalar::to<T>() { \
  return to##name(); \
}
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_TO)
#undef DEFINE_TO
}
