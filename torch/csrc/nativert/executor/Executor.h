#pragma once

#include <atomic>
#include <memory>

#include <c10/util/Logging.h>
#include <c10/util/Synchronized.h>

#include <torch/csrc/nativert/common/MPMCQueue.h>
#include <torch/csrc/nativert/common/Pytree.h>
#include <torch/csrc/nativert/common/Semaphore.h>
#include <torch/csrc/nativert/executor/ConstantFolder.h>
#include <torch/csrc/nativert/executor/DelegateExecutor.h>
#include <torch/csrc/nativert/executor/ExecutionPlanner.h>
#include <torch/csrc/nativert/executor/ExecutorConfig.h>
#include <torch/csrc/nativert/executor/GraphExecutorBase.h>
#include <torch/csrc/nativert/executor/Placement.h>
#include <torch/csrc/nativert/graph/Graph.h>
#include <torch/csrc/nativert/graph/GraphSignature.h>
#include <torch/csrc/nativert/kernels/KernelFactory.h>

namespace torch::nativert {

class Weights;
struct DistributedRunConfig;

/**
 * A very dumb executor. Basically just runs each node in order and contains a
 * giant unordered map for every intermediate, no optimizations applied.
 */
class Executor {
  class ExecutorFrameDeleter {
   public:
    explicit ExecutorFrameDeleter(Executor& e) : e_(&e) {}
    ExecutorFrameDeleter(ExecutorFrameDeleter&&) = default;
    ExecutorFrameDeleter& operator=(ExecutorFrameDeleter&&) = default;
    ExecutorFrameDeleter(const ExecutorFrameDeleter&) = default;
    ExecutorFrameDeleter& operator=(const ExecutorFrameDeleter&) = default;
    ~ExecutorFrameDeleter() = default;

    void operator()(ExecutionFrame* p) {
      e_->returnExecutorFrameToPool(std::unique_ptr<ExecutionFrame>(p));
    }

   private:
    Executor* e_;
  };
  class ExecutorFramePtr {
   public:
    ExecutorFramePtr(std::unique_ptr<ExecutionFrame> ptr, Executor& e)
        : ptr_(std::unique_ptr<ExecutionFrame, ExecutorFrameDeleter>(
              ptr.release(),
              ExecutorFrameDeleter{e})) {}
    ExecutorFramePtr() = delete;
    ExecutorFramePtr(ExecutorFramePtr&&) = default;
    ExecutorFramePtr& operator=(ExecutorFramePtr&&) = default;
    ExecutorFramePtr(const ExecutorFramePtr&) = delete;
    ExecutorFramePtr& operator=(const ExecutorFramePtr&) = delete;
    ~ExecutorFramePtr() = default;

    ExecutionFrame& operator*() {
      return *ptr_;
    }

    ExecutionFrame* operator->() {
      return ptr_.get();
    }

   private:
    std::unique_ptr<ExecutionFrame, ExecutorFrameDeleter> ptr_;
  };

 public:
  // Constrcutor used for Inference Path
  Executor(
      ExecutorConfig executorConfig,
      std::shared_ptr<Graph> graph,
      std::shared_ptr<Weights> weights,
      const Placement& placement = Placement(),
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader = nullptr);

  std::shared_ptr<Weights> getWeights() {
    std::shared_ptr<Weights> ret;
    weights_.withLock([&](auto& w) { ret = w; });
    return ret;
  }

  void processWeights(std::shared_ptr<Weights> weights);
  void atomicSwapWeights(std::shared_ptr<Weights> weights);

  // This API only returns the flattened UserOutputs,
  // intended to be used for Inference path
  // TODO Investigate whether we should remove this, still seems
  //      useful for testing.
  std::vector<c10::IValue> execute(std::vector<c10::IValue> inputs);

  std::vector<c10::IValue> execute(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const TreeSpec& inputTreeSpec);

  ProfileMetrics benchmarkIndividualNodes(
      std::vector<std::vector<c10::IValue>> inputsList,
      const uint32_t warmupRuns,
      const uint32_t mainRuns);

  const GraphSignature& graphSignature() const {
    return graph_->signature();
  }

  static std::string className() {
    return "Executor.v0";
  }

  const ExecutorConfig& executorConfig() const {
    return executorConfig_;
  }

  std::vector<DelegateExecutor*> getDelegates();

 protected:
  ExecutorConfig executorConfig_;

  std::shared_ptr<Graph> graph_;

  // manages the parameters, buffers and tensor constants
  c10::Synchronized<std::shared_ptr<Weights>> weights_;

  ExecutorFramePtr executeInternal(std::vector<c10::IValue> inputs);

  void initialize(
      std::shared_ptr<Weights> weights,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader);

  ExecutorFramePtr getExecutorFrameFromPool();
  void returnExecutorFrameToPool(std::unique_ptr<ExecutionFrame> frame);

 private:
  void maybeRunConstantFolding(std::shared_ptr<Weights> weights);
  void validateInputs(const std::vector<c10::IValue>& inputs) const;

  std::unique_ptr<GraphExecutorBase> graphExecutor_;

  const Placement placement_;

  // NOTE: delegateExecutors_ is used by nodeKernels_ inside graphExecutor_.
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors_;

  std::vector<ConstFoldingExecution> constFoldingExecutions_;

  std::optional<ConstantFolder> constantFolder_;

  Semaphore sem_;
  MPMCQueue<std::unique_ptr<ExecutionFrame>> executionFrames_;
  std::atomic_int numExecutionFrames_;
};

} // namespace torch::nativert
