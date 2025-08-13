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
  bool doExecutionFrameCleanup = true;
  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;
  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;
  int64_t minNumExecutionFrames = 1;
  int64_t executionFramePoolCleanupIntervalSec = 600;
  std::string modelName = "unknown";
};

} // namespace torch::nativert
