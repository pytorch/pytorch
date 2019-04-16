#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = kernel_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

}}
