#pragma once

#include "TensorLib/Scalar.h"
#include "TensorLib/Type.h"

namespace tlib {

class Type;
struct Tensor {
  Tensor(Type * type)
  : type_(type) {}
  Type & type() const {
    return *type_;
  }
  virtual const char * toString() const = 0;

  //example
  //Tensor * add(Tensor & b);
  ${tensor_method_declarations}

  virtual ~Tensor() {}

  friend class Type;
private:
  Type * type_;
};

}
