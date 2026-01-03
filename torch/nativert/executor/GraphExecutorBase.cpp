#include <ATen/record_function.h>
#include <torch/nativert/executor/GraphExecutorBase.h>

#include <c10/util/Logging.h>
#include <caffe2/core/timer.h>

namespace torch::nativert {

GraphExecutorBase::GraphExecutorBase(
    const Graph& graph,
    std::vector<std::unique_ptr<OpKernel>> nodeKernels,
    const ExecutorConfig& executorConfig)
    : graph_(graph),
      nodeKernels_(std::move(nodeKernels)),
      executorConfig_(executorConfig),
      execPlan_(ExecutionPlanner{graph_}.createPlan()) {}

void GraphExecutorBase::fillUserInputs(
    ExecutionFrame& frame,
    std::vector<c10::IValue> inputs) {
  RECORD_USER_SCOPE("Executor::fillUserInputs");
  const auto& inputValues = graph_.userInputs();
  TORCH_CHECK(inputValues.size() == inputs.size());

  // load user input tensor into execution frame
  for (size_t i = 0; i < inputValues.size(); i++) {
    if (inputValues[i]) {
      frame.setIValue(inputValues[i]->id(), std::move(inputs[i]));
    }
  }
}

ProfileMetrics GraphExecutorBase::benchmarkIndividualNodes(
    ExecutionFrame& executionFrame,
    const std::vector<std::vector<c10::IValue>>& inputsList,
    const uint32_t warmupRuns,
    const uint32_t mainRuns) {
  // TODO: add support for memory profiling
  TORCH_CHECK(warmupRuns >= 1 && mainRuns >= 1);

  ProfileMetrics results;
  const auto numNodes = static_cast<uint32_t>(nodeKernels_.size());

  results.percentPerNode.resize(numNodes, 0.0f);
  results.nodeTypes.reserve(numNodes);
  for (const auto& nodeKernel : nodeKernels_) {
    results.nodeTypes.emplace_back(nodeKernel->node()->target());
  }

  results.timePerNode.resize(numNodes, 0);
  if (inputsList.empty()) {
    auto i = 0;
    for (const auto& nodeKernel : nodeKernels_) {
      std::string target(nodeKernel->node()->target());
      results.timePerNode[i] = 0;
      results.timePerNodeType[target] = 0;
      results.instancesPerNodeType[target]++;
      if (nodeKernel->hasPrimKernel()) {
        results.primNodesCount++;
        results.primNodes.insert(target);
      } else if (nodeKernel->hasStaticDispatch()) {
        results.staticDispatchNodesCount++;
        results.staticDispatchNodes.insert(target);
      }
      i++;
    }
    results.totalNodesCount = numNodes;
    for (const auto& p : results.timePerNodeType) {
      const std::string& kind = p.first;
      results.percentPerNodeType[kind] = 0;
    }
    return results;
  }

  // Warmup
  for (uint32_t i = 0; i < warmupRuns; i++) {
    for (const auto& inputs : inputsList) {
      execute(executionFrame, inputs);
    }
  }

  // Execute kernels
  caffe2::Timer timer;
  executionFrame.withManagedMemory([&](auto) {
    for (uint32_t i = 0; i < mainRuns; i++) {
      for (auto inputs : inputsList) {
        const auto& inputValues = graph_.userInputs();

        TORCH_CHECK(inputValues.size() == inputs.size());
        for (size_t j = 0; j < inputValues.size(); j++) {
          executionFrame.setIValue(inputValues[j]->id(), std::move(inputs[j]));
        }
        for (NodeIndex nodeIdx = 0; nodeIdx < nodeKernels_.size(); ++nodeIdx) {
          timer.Start();
          nodeKernels_[nodeIdx]->compute(executionFrame);
          float millis = timer.MilliSeconds();
          results.timePerNode[nodeIdx] += millis;
        }
      }
    }
  });

  // Summarize results
  const float numTotalIters =
      (static_cast<float>(mainRuns) * static_cast<float>(inputsList.size()));
  for (const auto i : c10::irange(numNodes)) {
    const Node* node = nodeKernels_[i]->node();
    std::string target(node->target());
    results.timePerNode[i] /= numTotalIters;
    results.timePerNodeType[target] += results.timePerNode[i];
    results.instancesPerNodeType[target]++;
    if (nodeKernels_[i]->hasPrimKernel()) {
      results.primNodes.insert(target);
      results.primNodesCount++;
    } else if (nodeKernels_[i]->hasStaticDispatch()) {
      results.staticDispatchNodes.insert(target);
      results.staticDispatchNodesCount++;
    }
    results.totalTime += results.timePerNode[i];
  }
  results.totalNodesCount = numNodes;
  for (const auto& r : results.timePerNodeType) {
    const std::string& target = r.first;
    results.percentPerNodeType[target] = r.second * 100.0f / results.totalTime;
  }
  for (const auto i : c10::irange(numNodes)) {
    results.percentPerNode[i] =
        results.timePerNode[i] * 100.0f / results.totalTime;
  }
  return results;
}

} // namespace torch::nativert
