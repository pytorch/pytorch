#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>
#include <c10/util/OptionalArrayRef.h>

namespace at {

class Tensor;

namespace native {

at::Tensor _set_data(at::Tensor& a, at::Tensor const& b) {
  a.set_data(b);
  return a;
}

}
}  // namespace at::native