#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>

namespace at {

class Tensor;

namespace native {

template<typename O, typename C>
void _assert_match(const O& original, const C& compared, const std::string& name) {
  if (compared) {
    std::stringstream msg;
    bool equal = (original == compared.value());
    if (!equal) {
      msg << "Tensor " << name << " mismatch!";
      AT_ASSERT(euql, msg.str());
    }
  }
}

void _assert_tensor_metadata(at::Tensor const& tensor, c10::OptionalArrayRef<long> sizes, c10::OptionalArrayRef<long> strides, c10::optional<c10::ScalarType> dtype) {
  // std::stringstream msg;
  // if (sizes) {
  //   bool size_match = (tensor.sizes() == sizes.value())
  //   if (!size_match) {
  //     msg << "Tensor size mismatch! Got " << tensor.sizes().vec() << " expected" << sizes.value().vec() << std::endl;
  //     AT_ASSERT(size_match, msg.str());
  //   }
  // }
  // if (strides) {
  //   bool strides_match = (tensor.strides() == strides.value())
  //   if (!strides_match) {
  //     msg << "Tensor stride mismatch! Got " << tensor.strides() << " expected" << strides.value() << std::endl;
  //     AT_ASSERT(tensor.strides() == strides.value(), msg.str());
  //   }
  // }
  // if (dtype) {
  //   msg << "Tensor dtype mismatch! Got " << tensor.dtype() << " expected" << dtype.value() << std::endl;
  //   AT_ASSERT(tensor.dtype() == dtype.value(), msg.str());
  // }
}

}
}  // namespace at::native
