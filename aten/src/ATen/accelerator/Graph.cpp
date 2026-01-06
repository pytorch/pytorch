#include <ATen/DeviceAccelerator.h>
#include <ATen/accelerator/Graph.h>

namespace at::accelerator {

Graph::Graph(bool keep_graph) {
  c10::DeviceType device_type = at::accelerator::getAccelerator(true).value();
  TORCH_CHECK(
      has_graph_impl(device_type),
      "Graph is not supported on device type: ",
      device_type);
  GraphImplArgs args{keep_graph};
  impl_ = create_graph_impl(device_type, args);
}

bool isGraphAvailable() {
  c10::DeviceType device_type = at::accelerator::getAccelerator(true).value();
  return has_graph_impl(device_type);
}

} // namespace at::accelerator
