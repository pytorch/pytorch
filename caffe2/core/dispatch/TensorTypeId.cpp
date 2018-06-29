#include "caffe2/core/dispatch/TensorTypeId.h"

std::ostream& operator<<(std::ostream& str, c10::TensorTypeId rhs) {
  return str << rhs.underlyingId();
}
