#pragma once

#include <chrono>
#include <memory>

#include <c10/util/Registry.h>

namespace c10d::cuda {

enum StreamBlockStatus : int32_t {
  UNKNOWN = 0,
  RUNNING = 1,
  TIMED_OUT = 2,
  ABORTED = 3,
};

/*
StreamBlock implements a baton that will block a the active CUDA stream
until aborted by the main process.
*/
class TORCH_API StreamBlock {
 public:
  virtual ~StreamBlock() = default;
  virtual void abort() = 0;
  virtual StreamBlockStatus status() = 0;
};

std::unique_ptr<StreamBlock> block_stream(std::chrono::milliseconds timeout);

// Declare a registry so we can call the CUDA StreamBlock API from CPU only code
// (i.e. ProcessGroup/Work objects in libtorch_cpu).
// The implementation lives defined in StreamBlock.cu.
TORCH_DECLARE_REGISTRY(
    StreamBlockRegistry,
    StreamBlock,
    std::chrono::milliseconds);

} // namespace c10d::cuda
