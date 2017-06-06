#pragma once

namespace tlib {

class Type;
struct Tensor {
  virtual Type & type() const = 0;


  Tensor * add(Tensor & b);
  ${tensor_method_declarations}
  
  virtual ~Tensor() {}
};

}
