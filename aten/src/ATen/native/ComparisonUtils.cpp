#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>

namespace at {

class Tensor;

namespace native {

void _assert_tensor_metadata(at::Tensor const& tensor, c10::OptionalArrayRef<long> sizes, c10::OptionalArrayRef<long> strides, c10::optional<c10::ScalarType> dtype) {
  std::stringstream msg;
  if (sizes) {
    msg << "Tensor size mismatch! Got " << tensor.sizes().vec() << " expected" << sizes.value().vec() << std::endl;
    AT_ASSERT(tensor.sizes() == sizes.value(), msg.str());
  }
  if (strides) {
    msg << "Tensor stride mismatch! Got " << tensor.strides() << " expected" << strides.value() << std::endl;
    AT_ASSERT(tensor.strides() == strides.value(), msg.str());
  }
  if (dtype) {
    msg << "Tensor dtype mismatch! Got " << tensor.dtype() << " expected" << dtype.value() << std::endl;
    AT_ASSERT(tensor.dtype() == dtype.value(), msg.str());
  }
}

}
}  // namespace at::native
