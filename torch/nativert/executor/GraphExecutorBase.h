#pragma once

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphSignature.h>

namespace torch::nativert {

struct ProfileMetrics {
  size_t primNodesCount{0};
  size_t staticDispatchNodesCount{0};
  size_t totalNodesCount{0};
  std::vector<float> timePerNode;
  std::vector<std::string> nodeTypes;
  std::unordered_map<std::string, float> timePerNodeType;
  std::unordered_map<std::string, float> percentPerNodeType;
  std::vector<float> percentPerNode;
  std::unordered_map<std::string, int> instancesPerNodeType;
  std::unordered_set<std::string> staticDispatchNodes;
  std::unordered_set<std::string> primNodes;
  float totalTime{0};
  std::string name;
};

/**
 * GraphExecutor is a lightweight abstraction to execute a graph with
 * execution frames without actually owning the graph nor the weights. This is
 * introduced to decouple the state management of the top level runtime from the
 * kernel executions so that sub graphs from higher order ops can be supported.
 */
class GraphExecutorBase {
 public:
  GraphExecutorBase(
      const Graph& graph,
      std::vector<std::unique_ptr<OpKernel>> nodeKernels,
      const ExecutorConfig& executorConfig);
  virtual ~GraphExecutorBase() = default;

  const Graph& graph() const {
    return graph_;
  }

  // This API only returns the flattened UserOutputs,
  // intended to be used for Inference path
  virtual std::vector<c10::IValue> execute(
      ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) = 0;

  virtual std::vector<c10::IValue> executeWithPrefilledFrame(
      ExecutionFrame& frame) = 0;

  ProfileMetrics benchmarkIndividualNodes(
      ExecutionFrame& executionFrame,
      const std::vector<std::vector<c10::IValue>>& inputs,
      const uint32_t warmup_runs,
      const uint32_t main_runs);

  std::vector<std::unique_ptr<OpKernel>> stealKernels() {
    return std::move(nodeKernels_);
  }

  void setKernels(std::vector<std::unique_ptr<OpKernel>>&& kernels) {
    nodeKernels_ = std::move(kernels);
  }

 protected:
  void fillUserInputs(ExecutionFrame& frame, std::vector<c10::IValue> inputs);

  const Graph& graph_;

  // cache of the constructed kernels to avoid reconstruction per execution
  std::vector<std::unique_ptr<OpKernel>> nodeKernels_;

  const ExecutorConfig& executorConfig_;

  std::unique_ptr<ExecutionPlan> execPlan_;
};

} // namespace torch::nativert
