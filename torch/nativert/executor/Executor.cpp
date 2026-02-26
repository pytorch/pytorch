#include <memory>

#include <c10/util/Enumerate.h>
#include <c10/util/Synchronized.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/executor/ParallelGraphExecutor.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/kernels/C10Kernel.h>
#include <torch/nativert/kernels/KernelFactory.h>

namespace torch::nativert {

Executor::Executor(
    torch::nativert::ExecutorConfig executorConfig,
    std::shared_ptr<Graph> graph,
    const std::shared_ptr<Weights>& weights,
    const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
        pytorchStreamReader)
    : executorConfig_(std::move(executorConfig)),
      graph_(std::move(graph)),
      constantFolder_(
          executorConfig_.runConstFolding
              ? std::optional<ConstantFolder>(*graph_)
              : std::nullopt),
      executionFrames_(executorConfig_.maxNumConcurrentThreads),
      inactiveExecutionFrames_(executorConfig_.maxNumConcurrentThreads),
      numExecutionFrames_(0),
      lastClearedTimestamp_(getCurrentTimestampSeconds()) {
  if (weights) {
    initialize(weights, pytorchStreamReader);
  }
}

void Executor::initialize(
    const std::shared_ptr<Weights>& weights,
    const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
        pytorchStreamReader) {
  auto start = std::chrono::high_resolution_clock::now();

  auto executionKernels = KernelFactory().initializeNodeKernels(
      *graph_, weights, executorConfig_, pytorchStreamReader);

  if (constantFolder_.has_value()) {
    constantFolder_->unlinkConstants(executionKernels.nodeKernels);
  }

  const auto& kernelSchemas = getKernelSchemas(executionKernels.nodeKernels);

  if (executorConfig_.maxParallelOps > 1) {
    graphExecutor_ = std::make_unique<ParallelGraphExecutor>(
        *graph_, std::move(executionKernels.nodeKernels), executorConfig_);
  } else {
    graphExecutor_ = std::make_unique<torch::nativert::SerialGraphExecutor>(
        *graph_, std::move(executionKernels.nodeKernels), executorConfig_);
  }

  delegateExecutors_ = std::move(executionKernels.delegateExecutors);
  constFoldingExecutions_ = std::move(executionKernels.constFoldingExecutions);

  initWeights(weights);

  if (executorConfig_.layoutPlannerSettings.enabled()) {
    layoutPlanner_ = std::make_unique<LayoutPlanner>(
        *graph_,
        kernelSchemas,
        ExecutionFrame::getPersistentValueMask(*graph_, weights.get()),
        executorConfig_.layoutPlannerSettings);
  }

  auto end = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Initialization completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                   .count()
            << " ms";
}

/* static */ c10::
    FastMap<std::string /* target */, torch::nativert::FunctionSchema>
    Executor::getKernelSchemas(
        const std::vector<std::unique_ptr<OpKernel>>& kernels) {
  c10::FastMap<std::string, torch::nativert::FunctionSchema> output;
  for (const auto& kernel : kernels) {
    if (const auto* casted = dynamic_cast<C10Kernel*>(kernel.get()); casted) {
      output.insert({std::string(kernel->node()->target()), casted->schema()});
    }
  }
  return output;
}

void Executor::atomicSwapWeights(std::shared_ptr<Weights> weights) {
  weights_.withLock([&](auto& w) { w = std::move(weights); });

  // update weights in delegate executors
  for (auto& delegateExecutor : delegateExecutors_) {
    delegateExecutor->commitWeights();
  }
}

void Executor::maybeRunConstantFolding(
    const std::shared_ptr<Weights>& weights) {
  for (auto& execution : constFoldingExecutions_) {
    ExecutionFrame constFoldingFrame(execution.executor->graph());
    std::vector<c10::IValue> inputs;
    inputs.reserve(graph_->signature().inputsToWeights().size());
    for (const auto& [_, name] : graph_->signature().inputsToWeights()) {
      inputs.emplace_back(weights->at(name));
    }

    auto outputs = execution.executor->execute(constFoldingFrame, inputs);
    for (const auto& [idx, value] :
         c10::enumerate(execution.executor->graph().outputs())) {
      weights->updateFoldedConst(value->name(), outputs.at(idx));
    }
  }
  // runtime constant folding after the run_const_graph HOPs, if applicable
  if (constantFolder_.has_value()) {
    constantFolder_->evaluate(*weights);
  }
}

void Executor::processWeights(const std::shared_ptr<Weights>& weights) {
  maybeRunConstantFolding(weights);
  for (auto& delegateExecutor : delegateExecutors_) {
    delegateExecutor->processWeights(weights);
  }
}

void Executor::initWeights(const std::shared_ptr<Weights>& weights) {
  maybeRunConstantFolding(weights);

  weights_.withLock([&](auto& w) { w = std::move(weights); });

  for (auto& delegateExecutor : delegateExecutors_) {
    delegateExecutor->initWeights(weights);
  }
}

namespace {
void validateInput(
    const std::string& inputName,
    const at::Tensor& inputTensor,
    const torch::nativert::TensorMeta& tensorValueMeta) {
  TORCH_CHECK(
      inputTensor.dtype() == tensorValueMeta.dtype(),
      "Input tensor dtype mismatch for ",
      inputName,
      ", expecting ",
      c10::toString(tensorValueMeta.dtype()),
      " but got ",
      inputTensor.dtype().name());

  TORCH_CHECK(
      inputTensor.device() == tensorValueMeta.device(),
      "Input tensor device mismatch for ",
      inputName,
      ", expecting ",
      tensorValueMeta.device().str(),
      " but got ",
      inputTensor.device().str());
}

} // namespace

// validate input tensor's dtype matches tensorMeta
void Executor::validateInputs(const std::vector<c10::IValue>& inputs) const {
  const auto& inputValues = graph_->userInputs();
  const auto& tensorValuesMeta = graph_->tensorValuesMeta();
  TORCH_CHECK(inputs.size() == inputValues.size(), "Input size mismatch");
  for (auto&& [i, actualInput] : c10::enumerate(inputs)) {
    if (actualInput.isTensor()) {
      const auto& inputName = std::string(inputValues[i]->name());
      auto it = tensorValuesMeta.find(inputName);
      TORCH_CHECK(
          it != tensorValuesMeta.end(),
          "Couldn't find ",
          inputName,
          " in tensorValuesMeta");
      validateInput(inputName, actualInput.toTensor(), it->second);
    }
  }
}

Executor::ExecutorFramePtr Executor::getExecutorFrameFromPool() {
  std::shared_ptr<Weights> weights;
  weights_.withLock([&](auto& w) { weights = w; });

  // Try to get a frame from the main pool or create a new one
  std::unique_ptr<ExecutionFrame> frame;

  // Try to get a frame from executionFrames_ or inactiveExecutionFrames_
  while (!executionFrames_.readIfNotEmpty(frame) &&
         !inactiveExecutionFrames_.readIfNotEmpty(frame)) {
    int64_t numFrames = numExecutionFrames_.load();
    if (numFrames < executorConfig_.maxNumConcurrentThreads) {
      if (numExecutionFrames_.compare_exchange_strong(
              numFrames, numFrames + 1)) {
        return ExecutorFramePtr{
            std::make_unique<ExecutionFrame>(
                *graph_, *weights, executorConfig_, layoutPlanner_.get()),
            *this};
      }
    } else {
      sem_.acquire();
    }
  }
  ExecutorFramePtr ptr{std::move(frame), *this};

  if (ptr->weightVersion() != weights->version()) {
    ptr->setWeights(*weights);
  }
  return ptr;
}

void Executor::clearStaleExecutionFrames() {
  LOG(INFO) << "Clearing stale execution frames";
  if (!cleanupLock_.try_lock()) {
    // Another thread is already doing cleanup
    return;
  }
  // Update timestamp first to minimize contention
  lastClearedTimestamp_ = getCurrentTimestampSeconds();

  // Get the size of active execution frames queue directly
  size_t activeFramesSize = executionFrames_.size();
  size_t inactiveFramesSize = inactiveExecutionFrames_.size();
  size_t total = activeFramesSize + inactiveFramesSize;
  size_t numCleared = 0;
  std::unique_ptr<ExecutionFrame> frame;

  // If number of active frames is less than the configured min, then transfer
  // the difference from inactive frames
  size_t minFramesToKeep = std::min(
      static_cast<size_t>(executorConfig_.minNumExecutionFrames), total);
  size_t framesToTransfer =
      (minFramesToKeep - activeFramesSize) > minFramesToKeep
      ? static_cast<size_t>(0)
      : minFramesToKeep - activeFramesSize;
  ;
  for (size_t i = 0;
       i < framesToTransfer && inactiveExecutionFrames_.readIfNotEmpty(frame);
       ++i) {
    executionFrames_.writeIfNotFull(std::move(frame));
  }

  size_t newActiveFramesSize = executionFrames_.size();

  // Clear remaining inactive frames (i.e. those that were not used in the last
  // time interval)
  while (inactiveExecutionFrames_.readIfNotEmpty(frame)) {
    ++numCleared;
    frame.reset();
    numExecutionFrames_ -= 1;
  }

  // Move active frames to inactive so they are cleared next time if not used
  // Check  newActiveFramesSize > 0 to guuard against other threads adding
  // frames to active queue during while loop
  while (executionFrames_.readIfNotEmpty(frame) && newActiveFramesSize > 0) {
    --newActiveFramesSize;
    inactiveExecutionFrames_.writeIfNotFull(std::move(frame));
  }

  LOG(INFO) << "Cleared " << numCleared << " out of " << total
            << " ExecutionFrame instances in the pool";

  cleanupLock_.unlock();
}

void Executor::returnExecutorFrameToPool(
    std::unique_ptr<ExecutionFrame> frame) {
  // Check if it's time to clean up stale frames
  // TODO: consider moving cleanup to a dedicated thread so it does not impact
  // p99 latency
  if (executorConfig_.doExecutionFrameCleanup &&
      lastClearedTimestamp_ +
              executorConfig_.executionFramePoolCleanupIntervalSec <
          getCurrentTimestampSeconds()) {
    clearStaleExecutionFrames();
  }

  try {
    frame->destroyBorrowedIValues();
    // When tryFreeUnmanagedValuesAfterUse is false, intermediate values are
    // not freed during execution. Clear them here to prevent stale tensor
    // data from persisting across frame reuses (e.g. during rebatching with
    // varying batch sizes, stale tensors can cause out-of-bounds accesses).
    if (!executorConfig_.tryFreeUnmanagedValuesAfterUse) {
      frame->clearNonPersistentValues();
    }
    // Always return to active execution frame pool, indicating that frame was
    // used in the previous time interval
    TORCH_CHECK(
        executionFrames_.writeIfNotFull(std::move(frame)),
        "ExecutionFrame pool full");
  } catch (...) {
    sem_.release();
    throw;
  }
  sem_.release();
}

std::vector<c10::IValue> Executor::execute(std::vector<c10::IValue> inputs) {
  if (executorConfig_.validateInputs) {
    validateInputs(inputs);
  }

  auto executionFrame = getExecutorFrameFromPool();
  return graphExecutor_->execute(*executionFrame, std::move(inputs));
}

std::vector<c10::IValue> Executor::execute(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const ITreeSpec& inputTreeSpec) {
  auto executionFrame = getExecutorFrameFromPool();

  std::optional<std::vector<c10::IValue>> outputs;
  const auto userInputs = graph_->userInputs();
  const auto& tensorValuesMeta = graph_->tensorValuesMeta();
  TORCH_CHECK(userInputs.size() == inputTreeSpec.numIValues());

  auto executionFrameFillUserInputs = [&](const c10::IValue& leaf,
                                          const Value* value) {
    // validate input tensor's dtype and device matches tensorMeta
    if (executorConfig_.validateInputs && leaf.isTensor()) {
      const auto& inputName = std::string(value->name());
      auto it = tensorValuesMeta.find(inputName);
      TORCH_CHECK(
          it != tensorValuesMeta.end(),
          "Couldn't find ",
          inputName,
          " in tensorValuesMeta");
      validateInput(inputName, leaf.toTensor(), it->second);
    }
    executionFrame->setBorrowedIValue(
        value->id(), c10::MaybeOwnedTraits<c10::IValue>::createBorrow(leaf));
  };
  ivalueApplyFromArgs(
      executionFrameFillUserInputs, args, kwargs, inputTreeSpec);
  try {
    outputs = graphExecutor_->executeWithPrefilledFrame(*executionFrame);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during executeWithPrefilledFrame: " << e.what();
    throw;
  }

  return std::move(*outputs);
}

ProfileMetrics Executor::benchmarkIndividualNodes(
    const std::vector<std::vector<c10::IValue>>& inputsList,
    const uint32_t warmupRuns,
    const uint32_t mainRuns) {
  TORCH_CHECK(!inputsList.empty(), "Need at least one input to benchmark");
  TORCH_CHECK(warmupRuns >= 1 && mainRuns >= 1, "Need at least one run");

  for (const auto& inputs : inputsList) {
    if (executorConfig_.validateInputs) {
      validateInputs(inputs);
    }
  }
  auto executionFrame = getExecutorFrameFromPool();
  auto benchmarkResults = graphExecutor_->benchmarkIndividualNodes(
      *executionFrame, inputsList, warmupRuns, mainRuns);

  return benchmarkResults;
}

int64_t Executor::getCurrentTimestampSeconds() const {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<DelegateExecutor*> Executor::getDelegates() {
  std::vector<DelegateExecutor*> delegates;
  delegates.reserve(delegateExecutors_.size());
  for (const auto& delegateExecutor : delegateExecutors_) {
    delegates.emplace_back(delegateExecutor.get());
  }
  return delegates;
}

} // namespace torch::nativert
