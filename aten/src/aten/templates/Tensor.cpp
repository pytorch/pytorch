#include "Tensor.h"
#include "Type.h"

namespace tlib {

  Tensor * Tensor::add(Tensor & b) {
    return type().add(*this,b);
  }

}
