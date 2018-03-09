#include <ATen/ATen.h>

#include <iostream>

namespace at {

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

bool Tensor::isVariable() const noexcept {
  return type().is_variable();
}

} // namespace at
