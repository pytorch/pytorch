#include "TensorLib/Tensor.h"
#include "TensorLib/Type.h"

namespace tlib {

  Tensor * Tensor::add(Tensor & b) {
    return type().add(*this,b);
  }
  ${tensor_method_definitions}

}
