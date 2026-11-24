#include <ATen/DeviceAccelerator.h>
#include <ATen/accelerator/Graph.h>
#include <ATen/core/Tensor.h>

namespace at::accelerator {

Graph::Graph(bool keep_graph) : Graph(keep_graph, 8) {}

Graph::Graph(bool keep_graph, int64_t num_rng_states) {
  TORCH_CHECK(num_rng_states >= 0, "num_rng_states must be nonnegative.");
  c10::DeviceType device_type = at::accelerator::getAccelerator(true).value();
  TORCH_CHECK(
      has_graph_impl(device_type),
      "Graph is not supported on device type: ",
      device_type);
  GraphImplArgs args{keep_graph, num_rng_states};
  impl_ = create_graph_impl(device_type, args);
}

std::vector<at::Tensor> Graph::get_rng_state_buffers() const {
  return impl_->get_rng_state_buffers();
}

} // namespace at::accelerator
