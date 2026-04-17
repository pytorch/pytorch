#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.hpp>

namespace c10d::cuda {

C10_DEFINE_REGISTRY(StreamBlockRegistry, StreamBlock, std::chrono::milliseconds)

std::unique_ptr<StreamBlock> block_stream(std::chrono::milliseconds timeout) {
  auto baton = StreamBlockRegistry()->Create("CUDA", timeout);
  TORCH_CHECK(baton, "Failed to create StreamBlock");
  return baton;
}

} // namespace c10d::cuda
