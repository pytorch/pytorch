#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor& vulkan_copy_(Tensor& self, const Tensor& src);

} // namespace native
} // namespace at
