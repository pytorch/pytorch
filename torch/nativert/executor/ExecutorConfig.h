#pragma once

#include <torch/nativert/executor/memory/LayoutPlannerSettings.h>
#include <cstdint>
#include <string>

namespace torch::nativert {

struct ExecutorConfig {
  bool validateInputs = false;
  bool debugNan = false;
  bool enableStaticCPUKernels = false;
  bool runConstFolding = false;
  bool doExecutionFrameCleanup = true;
  bool tryFreeUnmanagedValuesAfterUse = true;
  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;
  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;
  int64_t minNumExecutionFrames = 1;
  int64_t executionFramePoolCleanupIntervalSec = 600;
  LayoutPlannerSettings layoutPlannerSettings;
  std::string modelName = "unknown";
};

} // namespace torch::nativert
