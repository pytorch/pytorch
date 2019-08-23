#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

void backward(const Tensor& self, const Tensor& gradient, bool keep_graph, bool create_graph) {
  AT_ERROR("backward is not implemented for Tensor");
}

void set_data(const Tensor& self, const Tensor& new_data) {
  AT_ERROR("set_data is not implemented for Tensor");
}

} // namespace native
} // namespace at
