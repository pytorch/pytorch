#pragma once

#include <memory>

#include "ATen/Scalar.h"
#include "ATen/ArrayRef.h"

namespace at {

class Context;
class Storage;
class Tensor;
class Generator;

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  AT_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  NumOptions
};

enum class Backend {
  CPU,
  CUDA,
  NumOptions
};

constexpr Backend kCPU = Backend::CPU;
constexpr Backend kCUDA = Backend::CUDA;

static inline const char * toString(Backend b) {
  switch(b) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
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
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef DEFINE_CASE
}

struct CPUTag {
  static constexpr Backend value = Backend::CPU;
};
struct CUDATag {
  static constexpr Backend value = Backend::CUDA;
};

enum class TypeID {
  ${type_ids}
  NumOptions
};


typedef ArrayRef<int64_t> IntList;

struct Type {
  Type(Context * context)
  : context(context) {}
  virtual ScalarType scalarType() = 0;
  virtual Backend backend() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
  static void registerAll(Context * context);
  virtual std::unique_ptr<Storage> storage() = 0;
  virtual std::unique_ptr<Storage> storage(size_t size) = 0;
  virtual std::unique_ptr<Generator> generator() = 0;
  virtual const char * toString() const = 0;
  Type & toBackend(Backend b);
  Type & toScalarType(ScalarType s);

  // contingious IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  virtual void copy(const Tensor & src, Tensor & dst) = 0;
  Tensor copy(const Tensor & src);

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  ${type_method_declarations}
protected:
  Context* context;
};


}
