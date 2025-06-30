#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/cuda/Baton.h>

namespace c10d::cuda {

C10_DEFINE_REGISTRY(BatonRegistry, Baton, std::chrono::milliseconds)

std::unique_ptr<Baton> baton(std::chrono::milliseconds timeout) {
  auto baton = BatonRegistry()->Create("CUDA", timeout);
  TORCH_CHECK(baton, "Failed to create baton");
  return baton;
}

} // namespace c10d::cuda
