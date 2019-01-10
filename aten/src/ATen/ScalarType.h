#pragma once

#include <stdint.h>

#include "ATen/ArrayRef.h"
#include "ATen/ATenGeneral.h"
#include "ATen/Half.h"

namespace at {

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(int16_t,Short,i) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(Half,Half,d) \
_(float,Float,d) \
_(double,Double,d)

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  AT_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  Undefined,
  NumOptions
};

enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  Undefined,
  NumOptions
};

constexpr Backend kCPU = Backend::CPU;
constexpr Backend kCUDA = Backend::CUDA;
constexpr Backend kSparseCPU = Backend::SparseCPU;
constexpr Backend kSparseCUDA = Backend::SparseCUDA;

static inline Backend toSparse(Backend b) {
  switch (b) {
    case Backend::CPU: return Backend::SparseCPU;
    case Backend::CUDA: return Backend::SparseCUDA;
    case Backend::SparseCPU: return Backend::SparseCPU;
    case Backend::SparseCUDA: return Backend::SparseCUDA;
    default: throw std::runtime_error("Unknown backend");
  }
}

static inline Backend toDense(Backend b) {
  switch (b) {
    case Backend::CPU: return Backend::CPU;
    case Backend::CUDA: return Backend::CUDA;
    case Backend::SparseCPU: return Backend::CPU;
    case Backend::SparseCUDA: return Backend::CUDA;
    default: throw std::runtime_error("Unknown backend");
  }
}

static inline const char * toString(Backend b) {
  switch(b) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
    case Backend::SparseCPU: return "SparseCPU";
    case Backend::SparseCUDA: return "SparseCUDA";
    default: return "UNKNOWN_BACKEND";
  }
}

#define DEFINE_CONSTANT(_,name,_2) \
constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char * toString(ScalarType t) {
#define DEFINE_CASE(_,name,_2) \
  case ScalarType:: name : return #name;

  switch(t) {
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
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

static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
#define u1 ScalarType::Byte
#define i1 ScalarType::Char
#define i2 ScalarType::Short
#define i4 ScalarType::Int
#define i8 ScalarType::Long
#define f2 ScalarType::Half
#define f4 ScalarType::Float
#define f8 ScalarType::Double
#define ud ScalarType::Undefined
  static constexpr ScalarType _promoteTypesLookup
      [static_cast<int>(ScalarType::NumOptions)]
      [static_cast<int>(ScalarType::NumOptions)] = {
            /* u1  i1  i2  i4  i8  f2  f4  f8, ud */
    /* u1 */ { u1, i2, i2, i4, i8, f2, f4, f8, ud },
    /* i1 */ { i2, i1, i2, i4, i8, f2, f4, f8, ud },
    /* i2 */ { i2, i2, i2, i4, i8, f4, f4, f8, ud },
    /* i4 */ { i4, i4, i4, i4, i8, f8, f4, f8, ud },
    /* i8 */ { i8, i8, i8, i8, i8, f8, f4, f8, ud },
    /* f2 */ { f2, f2, f4, f8, f8, f2, f4, f8, ud },
    /* f4 */ { f4, f4, f4, f4, f4, f4, f4, f8, ud },
    /* f8 */ { f8, f8, f8, f8, f8, f8, f8, f8, ud },
    /* ud */ { ud, ud, ud, ud, ud, ud, ud, ud, ud },
  };
#undef u1
#undef i1
#undef i2
#undef i4
#undef i8
#undef f2
#undef f4
#undef f8
#undef ud
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

struct Tensor;
typedef ArrayRef<int64_t> IntList;
typedef ArrayRef<Tensor> TensorList;

} // namespace at
