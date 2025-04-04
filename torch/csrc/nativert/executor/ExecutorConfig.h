#pragma once

#include <torch/script.h>

namespace torch::nativert {

struct ExecutorConfig {
  bool validateInputs = false;

  bool debugNan = false;

  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;

  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;

  bool enableStaticCPUKernels = false;

  bool enableStaticMemoryPlanning = false;

  std::string modelName = "unknown";

  bool runConstFolding = false;
};

} // namespace torch::nativert
