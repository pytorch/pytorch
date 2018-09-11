#include <ATen/Tensor.h>
#include <ATen/Type.h>
#include <ATen/Formatting.h>

#include <iostream>

namespace at {

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

const char * Tensor::toString() const {
  return type().toString();
}

} // namespace at
