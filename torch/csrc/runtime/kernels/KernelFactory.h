#pragma once

#include <memory>

#include <torch/script.h>

#include "torch/csrc/runtime/executor/ExecutorConfig.h"
#include "torch/csrc/runtime/executor/GraphExecutorBase.h"
#include "torch/csrc/runtime/executor/OpKernel.h"

namespace torch::runtime {

class DelegateExecutor;

struct ConstFoldingExecution {
  std::unique_ptr<GraphExecutorBase> executor;
};

struct ExecutionKernels {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;
};

class KernelFactory {
 public:
  explicit KernelFactory() {}

  ExecutionKernels initializeNodeKernels(
      const Graph& graph,
      std::shared_ptr<Weights> weights,
      const ExecutorConfig& executorConfig,
      const Placement& placement,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader = nullptr);
};
} // namespace torch::runtime
