#pragma once

#include "ATen/core/ArrayRef.h"
#include "ATen/core/Half.h"
#include "ATen/core/typeid.h"

#include <cstdint>
#include <iostream>
#include <complex>

namespace at {

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
_(uint8_t,Byte,i)  /* 0 */ \
_(int8_t,Char,i)   /* 1 */ \
_(int16_t,Short,i) /* 2 */ \
_(int,Int,i)       /* 3 */ \
_(int64_t,Long,i)  /* 4 */ \
_(at::Half,Half,d) /* 5 */ \
_(float,Float,d)   /* 6 */ \
_(double,Double,d) /* 7 */ \
_(at::ComplexHalf,ComplexHalf,z)        /* 8 */ \
_(std::complex<float>,ComplexFloat,z)   /* 9 */ \
_(std::complex<double>,ComplexDouble,z) /* 10 */

// If you want to support ComplexHalf for real, replace occurrences
// of this macro with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX.  But
// beware: convert() doesn't work for all the conversions you need...
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(_) \
_(uint8_t,Byte,i)  \
_(int8_t,Char,i)   \
_(int16_t,Short,i) \
_(int,Int,i)       \
_(int64_t,Long,i)  \
_(at::Half,Half,d) \
_(float,Float,d)   \
_(double,Double,d) \
_(std::complex<float>,ComplexFloat,z) \
_(std::complex<double>,ComplexDouble,z)

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i)  \
_(int8_t,Char,i)   \
_(int16_t,Short,i) \
_(int,Int,i)       \
_(int64_t,Long,i)  \
_(at::Half,Half,d) \
_(float,Float,d)   \
_(double,Double,d)

#define AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(int16_t,Short,i) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(float,Float,d) \
_(double,Double,d)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ENUM)
#undef DEFINE_ENUM
  Undefined,
  NumOptions
};

static inline DataType scalarTypeToDataType(ScalarType scalar_type) {
#define DEFINE_CASE(ctype, name, _) \
  case ScalarType::name:            \
    return caffe2::TypeIdentifier::Get<ctype>();

  switch(scalar_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    case ScalarType::Undefined: return DataType::uninitialized();
    default: AT_ERROR("Unrecognized Scalartype ", scalar_type, " (please report this error)");
  }
#undef DEFINE_CASE
}

static inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
#define DEFINE_CASE(ctype,name,_) \
  case ScalarType:: name : return caffe2::TypeMeta::Make<ctype>();

  switch(scalar_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    case ScalarType::Undefined: return caffe2::TypeMeta();
    default: AT_ERROR("Unrecognized Scalartype ", scalar_type, " (please report this error)");
  }
#undef DEFINE_CASE
}

static inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
#define DEFINE_IF(ctype, name, _)                      \
  if (dtype == caffe2::TypeMeta::Make<ctype>()) { \
    return ScalarType::name;                           \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_IF)
#undef DEFINE_IF
  if (dtype == caffe2::TypeMeta()) {
    return ScalarType::Undefined;
  }
  AT_ERROR("Unsupported TypeMeta in ATen: ", dtype, " (please report this error)");
}

static inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  return typeMetaToScalarType(m) == t;
}

static inline bool operator==(caffe2::TypeMeta m, ScalarType t) {
  return typeMetaToScalarType(m) == t;
}

#define DEFINE_CONSTANT(_,name,_2) \
constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char * toString(ScalarType t) {
#define DEFINE_CASE(_,name,_2) \
  case ScalarType:: name : return #name;

  switch(t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype,name,_2) \
  case ScalarType:: name : return sizeof(ctype);

  switch(t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CASE_ELEMENTSIZE_CASE)
    default:
      AT_ERROR("Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

static inline bool isIntegralType(ScalarType t) {
  return (t == ScalarType::Byte ||
          t == ScalarType::Char ||
          t == ScalarType::Int ||
          t == ScalarType::Long ||
          t == ScalarType::Short);
}

static inline bool isFloatingType(ScalarType t) {
  return (t == ScalarType::Double ||
          t == ScalarType::Float ||
          t == ScalarType::Half);
}

static inline bool isComplexType(ScalarType t) {
  return (t == ScalarType::ComplexHalf ||
          t == ScalarType::ComplexFloat ||
          t == ScalarType::ComplexDouble);
}

static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f2 = ScalarType::Half;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }
  if (isComplexType(a) || isComplexType(b)) {
    AT_ERROR("promoteTypes with complex numbers is not handled yet; figure out what the correct rules should be");
  }
  static constexpr ScalarType _promoteTypesLookup
      [static_cast<int>(ScalarType::NumOptions)]
      [static_cast<int>(ScalarType::NumOptions)] = {
            /* u1  i1  i2  i4  i8  f2  f4  f8 */
    /* u1 */ { u1, i2, i2, i4, i8, f2, f4, f8 },
    /* i1 */ { i2, i1, i2, i4, i8, f2, f4, f8 },
    /* i2 */ { i2, i2, i2, i4, i8, f2, f4, f8 },
    /* i4 */ { i4, i4, i4, i4, i8, f2, f4, f8 },
    /* i8 */ { i8, i8, i8, i8, i8, f2, f4, f8 },
    /* f2 */ { f2, f2, f2, f2, f2, f2, f4, f8 },
    /* f4 */ { f4, f4, f4, f4, f4, f4, f4, f8 },
    /* f8 */ { f8, f8, f8, f8, f8, f8, f8, f8 },
  };
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

class Tensor;
typedef ArrayRef<Tensor> TensorList;

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << at::toString(scalar_type);
}

} // namespace at
