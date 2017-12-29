#pragma once

#include <stdint.h>

#include "ATen/ArrayRef.h"
#include "ATen/ATenGeneral.h"
#include "ATen/Half.h"

namespace at {

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(double,Double,d) \
_(float,Float,d) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(int16_t,Short,i) \
_(Half,Half,d)

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

struct Tensor;
typedef ArrayRef<int64_t> IntList;
typedef ArrayRef<Tensor> TensorList;

} // namespace at
