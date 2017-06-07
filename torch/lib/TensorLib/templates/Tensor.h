#pragma once

#include "TensorLib/Scalar.h"
#include "TensorLib/Type.h"

namespace tlib {

class Type;
struct Tensor {
  virtual Type & type() const = 0;
  virtual const char * toString() const = 0;

  //example
  //Tensor * add(Tensor & b);
  ${tensor_method_declarations}

  virtual ~Tensor() {}

  friend class Type;
};

}
