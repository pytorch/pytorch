#pragma once

#include "TensorLib/Scalar.h"
#include "TensorLib/ArrayRef.h"

namespace tlib {

class Context;
class Storage;
class Tensor;
class Generator;

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  TLIB_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  NumOptions
};

enum class Processor {
  CPU,
  CUDA,
  NumOptions
};

typedef ArrayRef<int64_t> IntList;

struct Type {
  virtual ScalarType scalarType() = 0;
  virtual Processor processor() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
  static void registerAll(Context * context);
  virtual Storage * newStorage() = 0;
  virtual Storage * newStorage(size_t size) = 0;
  virtual Generator * newGenerator() = 0;
  virtual const char * toString() const = 0;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;

  ${type_method_declarations}

};


}
