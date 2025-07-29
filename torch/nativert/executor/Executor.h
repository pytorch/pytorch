#pragma once

#include <atomic>
#include <memory>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>
#include <c10/util/Semaphore.h>
#include <c10/util/Synchronized.h>

#include <torch/nativert/detail/ITree.h>
#include <torch/nativert/detail/MPMCQueue.h>
#include <torch/nativert/executor/ConstantFolder.h>
#include <torch/nativert/executor/DelegateExecutor.h>
#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>
#include <torch/nativert/executor/memory/LayoutPlanner.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphSignature.h>
#include <torch/nativert/kernels/KernelFactory.h>

namespace torch::nativert {

using namespace torch::nativert::detail;

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
  // Constructor used for Inference Path
  Executor(
      torch::nativert::ExecutorConfig executorConfig,
      std::shared_ptr<Graph> graph,
      const std::shared_ptr<Weights>& weights,
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          pytorchStreamReader = nullptr);

  std::shared_ptr<Weights> getWeights() {
    std::shared_ptr<Weights> ret;
    weights_.withLock([&](auto& w) { ret = w; });
    return ret;
  }

  void processWeights(const std::shared_ptr<Weights>& weights);
  void atomicSwapWeights(std::shared_ptr<Weights> weights);

  // This API only returns the flattened UserOutputs,
  // intended to be used for Inference path
  // TODO Investigate whether we should remove this, still seems
  //      useful for testing.
  std::vector<c10::IValue> execute(std::vector<c10::IValue> inputs);

  std::vector<c10::IValue> execute(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const ITreeSpec& inputTreeSpec);

  ProfileMetrics benchmarkIndividualNodes(
      const std::vector<std::vector<c10::IValue>>& inputsList,
      const uint32_t warmupRuns,
      const uint32_t mainRuns);

  const torch::nativert::GraphSignature& graphSignature() const {
    return graph_->signature();
  }

  static std::string className() {
    return "Executor.v0";
  }

  const torch::nativert::ExecutorConfig& executorConfig() const {
    return executorConfig_;
  }

  std::vector<DelegateExecutor*> getDelegates();

  // Get the number of execution frames in the pool
  int getNumExecutionFrames() const {
    return numExecutionFrames_.load();
  }

  static c10::FastMap<std::string /* target */, torch::nativert::FunctionSchema>
  getKernelSchemas(const std::vector<std::unique_ptr<OpKernel>>& kernels);

 protected:
  torch::nativert::ExecutorConfig executorConfig_;

  std::shared_ptr<Graph> graph_;

  // manages the parameters, buffers and tensor constants
  c10::Synchronized<std::shared_ptr<Weights>> weights_;

  void initialize(
      const std::shared_ptr<Weights>& weights,
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          pytorchStreamReader);

  ExecutorFramePtr getExecutorFrameFromPool();
  void returnExecutorFrameToPool(std::unique_ptr<ExecutionFrame> frame);

  // Clears stale execution frames from the pool
  void clearStaleExecutionFrames();

 private:
  // Structure to track execution frame usage
  struct ExecutionFrameEntry {
    bool used{false};
    std::unique_ptr<ExecutionFrame> frame;

    // Add move constructor and assignment operator
    ExecutionFrameEntry() = default;
    ExecutionFrameEntry(ExecutionFrameEntry&& other) noexcept
        : used(other.used), frame(std::move(other.frame)) {}
    ExecutionFrameEntry& operator=(ExecutionFrameEntry&& other) noexcept {
      used = other.used;
      frame = std::move(other.frame);
      return *this;
    }
    // Delete copy constructor and assignment operator
    ExecutionFrameEntry(const ExecutionFrameEntry&) = delete;
    ExecutionFrameEntry& operator=(const ExecutionFrameEntry&) = delete;
  };

  void maybeRunConstantFolding(const std::shared_ptr<Weights>& weights);
  void validateInputs(const std::vector<c10::IValue>& inputs) const;

  // Helper method to get current timestamp in seconds
  int64_t getCurrentTimestampSeconds() const;

  void initWeights(const std::shared_ptr<Weights>& weights);

  std::unique_ptr<GraphExecutorBase> graphExecutor_;

  // NOTE: delegateExecutors_ is used by nodeKernels_ inside graphExecutor_.
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors_;

  std::vector<ConstFoldingExecution> constFoldingExecutions_;

  std::optional<ConstantFolder> constantFolder_;

  c10::Semaphore sem_;
  torch::nativert::detail::MPMCQueue<std::unique_ptr<ExecutionFrame>>
      executionFrames_;
  torch::nativert::detail::MPMCQueue<ExecutionFrameEntry>
      clearedExecutionFrames_;
  std::atomic_int64_t numExecutionFrames_;

  std::unique_ptr<LayoutPlanner> layoutPlanner_;
  std::atomic_int64_t lastClearedTimestamp_;
  std::mutex cleanupLock_;
  std::atomic_bool clearingInProgress_{false};
};

} // namespace torch::nativert
