#include <ATen/core/GraphImplInterface.h>
#include <ATen/core/Tensor.h>

namespace at {

C10_DEFINE_REGISTRY(GraphImplRegistry, GraphImplInterface, GraphImplArgs)

std::vector<at::Tensor> GraphImplInterface::get_rng_state_buffers() const {
  return {};
}

} // namespace at
