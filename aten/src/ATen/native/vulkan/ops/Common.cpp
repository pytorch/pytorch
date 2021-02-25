#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

bool broadcast_first_input(const vTensor& input1, const vTensor& input2) {
  if ((input2.extents().data[1u] > 1 && input1.extents().data[1u] == 1)||
      (input2.extents().data[2u] > 1 && input1.extents().data[2u] == 1)||
       input2.extents().data[0u] > input1.extents().data[0u]) {
    return true;
  }
  return false;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
