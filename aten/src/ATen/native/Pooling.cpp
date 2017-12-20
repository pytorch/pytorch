#include "ATen/ATen.h"
#include "ATen/Check.h"
#include "ATen/NativeFunctions.h"

#include <sstream>
#include <vector>


namespace at { namespace native {

static void check1d(const char* name, IntList x) {
  if (x.size() != 1) {
    std::ostringstream ss;
    ss << "max_pool1d() argument '" << name << "' should contain one int (got "
       << x.size() << ")";
    throw std::runtime_error(ss.str());
  }
}

std::tuple<Tensor,Tensor> max_pool1d(
    const Tensor & self, IntList kernel_size, IntList stride, IntList padding,
    IntList dilation, bool ceil_mode) {

  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDim("max_pool1d", TensorArg(self, "self", 1), 3);
  check1d("kernel_size", kernel_size);
  check1d("stride", stride);
  check1d("padding", padding);
  check1d("dilation", dilation);

  Tensor output, indices;
  std::tie(output, indices) = at::max_pool2d(
      self.unsqueeze(2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

  return std::make_tuple(output.squeeze(2), indices.squeeze(2));
}

}}  // namespace at::native
