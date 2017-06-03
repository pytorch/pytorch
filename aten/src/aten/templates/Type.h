#pragma once

#include "Scalar.h"

namespace tlib {

class Context;
class Storage;

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

struct Type {
  virtual ScalarType scalarType() = 0;
  virtual Processor processor() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
  static void registerAll(Context * context);
  virtual Storage * newStorage() = 0;
  virtual Storage * newStorage(size_t size) = 0;
};

}
