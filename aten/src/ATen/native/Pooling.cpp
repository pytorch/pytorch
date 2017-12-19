#include "ATen/ATen.h"
#include "ATen/Check.h"
#include "ATen/NativeFunctions.h"

#include <sstream>
#include <vector>


namespace at { namespace native {

static std::vector<int64_t> pad(const char* name, IntList x, int64_t value) {
  if (x.size() != 1) {
    std::ostringstream ss;
    ss << "max_pool1d() argument '" << name << "' should contain one int (got "
       << x.size() << ")";
    throw std::runtime_error(ss.str());
  }
  auto v = std::vector<int64_t>(x);
  v.insert(v.begin(), value);
  return v;
}

std::tuple<Tensor,Tensor> max_pool1d(
    const Tensor & self, IntList kernel_size, IntList stride, IntList padding,
    IntList dilation, bool ceil_mode) {

  checkDimRange("max_pool1d", TensorArg(self, "self", 1), 2, 4);
  if (stride.empty()) {
    stride = kernel_size;
  }

  Tensor output, indices;
  std::tie(output, indices) = at::max_pool2d(
      self.unsqueeze(2),
      pad("kernel_size", kernel_size, 1),
      pad("stride", stride, 1),
      pad("padding", padding, 0),
      pad("dilation", dilation, 1),
      ceil_mode);

  return std::make_tuple(output.squeeze(2), indices.squeeze(2));
}

}}  // namespace at::native
