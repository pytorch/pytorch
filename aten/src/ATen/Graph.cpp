#include <ATen/Graph.h>
#include <ATen/DeviceAccelerator.h>

namespace at::accelerator {

Graph::Graph() {
  device_type_ = at::accelerator::getAccelerator(true).value();
  TORCH_CHECK(has_graph_impl(device_type_),
              "Graph is not supported on device type: ", device_type_);
  impl_ = create_graph_impl(device_type_);
}

} // namespace at::accelerator
