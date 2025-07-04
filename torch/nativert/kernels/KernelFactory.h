#pragma once

#include <memory>

#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/nativert/executor/DelegateExecutor.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

struct ConstFoldingExecution {
  std::unique_ptr<GraphExecutorBase> executor;
};

struct ExecutionKernels {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;
};

class KernelFactoryHandler {
 public:
  using OpKernelPtr = std::unique_ptr<OpKernel>;
  using DelegateExecutorPtr = std::unique_ptr<DelegateExecutor>;
  using Matcher = c10::function_ref<bool(
      const Node& node,
      const torch::nativert::ExecutorConfig&,
      c10::Device)>;
  using Callback =
      c10::function_ref<std::pair<OpKernelPtr, DelegateExecutorPtr>(
          const Node&,
          std::shared_ptr<Weights> weights,
          const torch::nativert::ExecutorConfig& executorConfig,
          caffe2::serialize::PyTorchStreamReader* pytorchStreamReader,
          c10::Device targetDevice)>;

  KernelFactoryHandler(Matcher matcher, Callback callback)
      : matcher_(matcher), callback_(callback) {}

  KernelFactoryHandler() = delete;
  KernelFactoryHandler(const KernelFactoryHandler&) = default;
  KernelFactoryHandler& operator=(const KernelFactoryHandler&) = default;
  KernelFactoryHandler(KernelFactoryHandler&&) = default;
  KernelFactoryHandler& operator=(KernelFactoryHandler&&) = default;
  ~KernelFactoryHandler() = default;

  bool match(
      const Node& node,
      const torch::nativert::ExecutorConfig& config,
      c10::Device device) const {
    return matcher_(node, config, device);
  }

  std::pair<OpKernelPtr, DelegateExecutorPtr> operator()(
      const Node& node,
      std::shared_ptr<Weights> weights,
      const torch::nativert::ExecutorConfig& executorConfig,
      caffe2::serialize::PyTorchStreamReader* pytorchStreamReader,
      c10::Device targetDevice) const {
    return callback_(
        node, weights, executorConfig, pytorchStreamReader, targetDevice);
  }

 private:
  Matcher matcher_;
  Callback callback_;
};

class KernelFactory {
 public:
  explicit KernelFactory() {}

  ExecutionKernels initializeNodeKernels(
      const Graph& graph,
      std::shared_ptr<Weights> weights,
      const torch::nativert::ExecutorConfig& executorConfig,
      const Placement& placement,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader = nullptr,
      const MakeProxyExecutorFn& makeProxyExecutorFunc = nullptr);

  static void registerHandler(
      const std::string& name,
      KernelFactoryHandler handler);
};

} // namespace torch::nativert
