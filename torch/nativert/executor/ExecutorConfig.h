#pragma once

#include <cstdint>
#include <string>

namespace torch::nativert {

struct ExecutorConfig {
  bool validateInputs = false;
  bool debugNan = false;
  bool enableStaticCPUKernels = false;
  bool enableStaticMemoryPlanning = false;
  bool runConstFolding = false;
  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;
  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;
  std::string modelName = "unknown";
};

} // namespace torch::nativert
