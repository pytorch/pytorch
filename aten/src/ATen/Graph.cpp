#include <ATen/DeviceAccelerator.h>
#include <ATen/Graph.h>

namespace at::accelerator {

Graph::Graph(bool keep_graph)
    : device_type_{at::accelerator::getAccelerator(true).value()} {
  TORCH_CHECK(
      has_graph_impl(device_type_),
      "Graph is not supported on device type: ",
      device_type_);
  GraphImplArgs args{
      .keep_graph = keep_graph,
  };
  impl_ = create_graph_impl(device_type_, args);
}

} // namespace at::accelerator
