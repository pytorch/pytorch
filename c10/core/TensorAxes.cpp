#include <c10/core/TensorAxes.h>

#include <iostream>

namespace c10 {

std::string TensorAxes::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::string toString(const TensorAxes& axes) {
  return axes.toString();
}

std::ostream& operator<<(std::ostream& stream, const TensorAxes& axes) {
  return stream << "TensorAxes(dtype=" << axes.dtype()
                << ", device=" << axes.device() << ", layout=" << axes.layout()
                << ", is_variable=" << axes.is_variable() << ")";
}

} // namespace c10
