#pragma once

#include "Scalar.h"

namespace tlib {

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  TLIB_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

enum class Processor {
  CPU,
  CUDA
};

class Type {
  virtual ScalarType scalarType() = 0;
  virtual Processor processor() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
};

}
