#include <c10d/DummyProcessGroupBackend.hpp>
#include <iostream>

namespace c10d {

c10::intrusive_ptr<Work> DummyProcessGroupBackend::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  std::cout << "in DummyProcessGroupBackend::broadcast" << std::endl;
  return c10::make_intrusive<Work>();
}

} // namespace c10d
