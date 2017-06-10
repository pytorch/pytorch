#pragma once

#include <memory>

#include "TensorLib/Scalar.h"
#include "TensorLib/ArrayRef.h"

namespace tlib {

class Context;
class Storage;
class Tensor;
class TensorRef;
class Generator;

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  TLIB_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  NumOptions
};

enum class Backend {
  CPU,
  CUDA,
  NumOptions
};

struct CPUTag {
  static constexpr Backend value = Backend::CPU;
};
struct CUDATag {
  static constexpr Backend value = Backend::CUDA;
};


typedef ArrayRef<int64_t> IntList;

struct Type {
  virtual ScalarType scalarType() = 0;
  virtual Backend backend() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
  static void registerAll(Context * context);
  virtual std::unique_ptr<Storage> newStorage() = 0;
  virtual std::unique_ptr<Storage> newStorage(size_t size) = 0;
  virtual std::unique_ptr<Generator> newGenerator() = 0;
  virtual const char * toString() const = 0;

  // contingious IDs for all types in the system
  // for external dispatch
  virtual int ID() const = 0;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;

  ${type_method_declarations}

};


}
