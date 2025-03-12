#include "torch/csrc/nativert/executor/Executor.h"

#include <memory>

#include <c10/util/Logging.h>

#include "torch/csrc/nativert/common/AutoTimer.h"
#include "torch/csrc/nativert/common/Enumerate.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/executor/ParallelGraphExecutor.h"
#include "torch/csrc/nativert/executor/SerialGraphExecutor.h"
#include "torch/csrc/nativert/executor/Weights.h"

namespace torch::nativert {

Executor::Executor(
    ExecutorConfig executorConfig,
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Weights> weights,
    const Placement& placement,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader)
    : executorConfig_(std::move(executorConfig)),
      graph_(std::move(graph)),
      placement_(placement),
      constantFolder_(
          executorConfig_.runConstFolding
              ? std::optional<ConstantFolder>(*graph_)
              : std::nullopt),
      executionFrames_(executorConfig_.maxNumConcurrentThreads),
      numExecutionFrames_(0) {
  if (weights) {
    initialize(std::move(weights), std::move(pytorchStreamReader));
  }
}

void Executor::initialize(
    std::shared_ptr<Weights> weights,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
        pytorchStreamReader) {
  AutoTimer t("Initialization completed");

  auto executionKernels = KernelFactory().initializeNodeKernels(
      *graph_,
      weights,
      executorConfig_,
      placement_,
      std::move(pytorchStreamReader));

  if (constantFolder_.has_value()) {
    constantFolder_->unlinkConstants(executionKernels.nodeKernels);
  }

  if (executorConfig_.maxParallelOps > 1) {
    graphExecutor_ = std::make_unique<ParallelGraphExecutor>(
        *graph_, std::move(executionKernels.nodeKernels), executorConfig_);
  } else {
    graphExecutor_ = std::make_unique<SerialGraphExecutor>(
        *graph_, std::move(executionKernels.nodeKernels), executorConfig_);
  }

  delegateExecutors_ = std::move(executionKernels.delegateExecutors);
  constFoldingExecutions_ = std::move(executionKernels.constFoldingExecutions);

  // initialize weights_
  processWeights(weights);
  atomicSwapWeights(std::move(weights));
}

void Executor::atomicSwapWeights(std::shared_ptr<Weights> weights) {
  weights_.withLock([&](auto& w) { w = std::move(weights); });

  // update weights in delegate executors
  for (auto& delegateExecutor : delegateExecutors_) {
    delegateExecutor->commitWeights();
  }
}

void Executor::maybeRunConstantFolding(std::shared_ptr<Weights> weights) {
  for (auto& execution : constFoldingExecutions_) {
    ExecutionFrame constFoldingFrame(execution.executor->graph());
    std::vector<c10::IValue> inputs;
    inputs.reserve(graph_->signature().inputsToWeights().size());
    for (const auto& [_, name] : graph_->signature().inputsToWeights()) {
      inputs.push_back(weights->at(name));
    }

    auto outputs = execution.executor->execute(constFoldingFrame, inputs);
    for (const auto& [idx, value] :
         enumerate(execution.executor->graph().outputs())) {
      weights->updateFoldedConst(value->name(), outputs.at(idx));
    }
  }
}

void Executor::processWeights(std::shared_ptr<Weights> weights) {
  maybeRunConstantFolding(weights);
  if (constantFolder_.has_value()) {
    constantFolder_->evaluate(*weights);
  }
  for (auto& delegateExecutor : delegateExecutors_) {
    delegateExecutor->processWeights(weights);
  }
}

namespace {

void validateInput(
    const std::string& inputName,
    const at::Tensor& inputTensor,
    const TensorMeta& tensorValueMeta) {
  CHECK(inputTensor.dtype() == tensorValueMeta.dtype())
      << "Input tensor dtype mismatch for " << inputName << ", expecting "
      << c10::toString(tensorValueMeta.dtype()) << " but got "
      << inputTensor.dtype().name();

  CHECK(inputTensor.device() == tensorValueMeta.device())
      << "Input tensor device mismatch for " << inputName << ", expecting "
      << tensorValueMeta.device().str() << " but got "
      << inputTensor.device().str();
}

} // namespace

// validate input tensor's dtype matches tensorMeta
void Executor::validateInputs(const std::vector<c10::IValue>& inputs) const {
  const auto& inputValues = graph_->userInputs();
  const auto& tensorValuesMeta = graph_->tensorValuesMeta();
  TORCH_CHECK(inputs.size() == inputValues.size(), "Input size mismatch");
  for (auto&& [i, actualInput] : enumerate(inputs)) {
    if (actualInput.isTensor()) {
      const auto& inputName = std::string(inputValues[i]->name());
      auto it = tensorValuesMeta.find(inputName);
      CHECK(it != tensorValuesMeta.end())
          << "Couldn't find " << inputName << " in tensorValuesMeta";
      validateInput(inputName, actualInput.toTensor(), it->second);
    }
  }
}

std::unique_ptr<ExecutionFrame> Executor::getExecutorFrameFromPool() {
  std::shared_ptr<Weights> weights;
  weights_.withLock([&](auto& w) { weights = w; });

  std::unique_ptr<ExecutionFrame> frame;
  while (!executionFrames_.readIfNotEmpty(frame)) {
    int numFrames = numExecutionFrames_.load();
    if (numFrames < executorConfig_.maxNumConcurrentThreads) {
      if (numExecutionFrames_.compare_exchange_strong(
              numFrames, numFrames + 1)) {
        return std::make_unique<ExecutionFrame>(*graph_, *weights);
      }
    } else {
      sem_.acquire();
    }
  }

  if (frame->weightVersion() != weights->version()) {
    frame->setWeights(*weights);
  }
  return frame;
}

void Executor::returnExecutorFrameToPool(
    std::unique_ptr<ExecutionFrame> frame) {
  if (executorConfig_.enableStaticCPUKernels) {
    frame->releaseUserOutputs();
  }
  CHECK(executionFrames_.writeIfNotFull(std::move(frame)))
      << "ExecutionFrame pool full";
  sem_.release();
}

std::unique_ptr<ExecutionFrame> Executor::executeInternal(
    std::vector<c10::IValue> inputs) {
  if (executorConfig_.validateInputs) {
    validateInputs(inputs);
  }

  auto executionFrame = getExecutorFrameFromPool();
  try {
    graphExecutor_->execute(*executionFrame, std::move(inputs));
  } catch (...) {
    returnExecutorFrameToPool(std::move(executionFrame));
    throw;
  }

  return executionFrame;
}

std::vector<c10::IValue> Executor::execute(std::vector<c10::IValue> inputs) {
  auto executionFrame = executeInternal(std::move(inputs));
  auto outputs = executionFrame->getUserOutputs();

  returnExecutorFrameToPool(std::move(executionFrame));

  return outputs;
}

std::vector<c10::IValue> Executor::execute(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const TreeSpec& inputTreeSpec) {
  auto executionFrame = getExecutorFrameFromPool();

  std::optional<std::vector<c10::IValue>> outputs;
  try {
    const auto& userInputs = graph_->userInputs();
    const auto& tensorValuesMeta = graph_->tensorValuesMeta();
    TORCH_CHECK_EQ(userInputs.size(), inputTreeSpec.numLeaves());

    size_t input_idx = 0;
    auto executionFrameFillUserInputs = [&](const c10::IValue& leaf) {
      auto value = userInputs[input_idx];
      // skip if the value is not used
      if (value && value->users().size() > 0) {
        // validate input tensor's dtype and device matches tensorMeta
        if (executorConfig_.validateInputs && leaf.isTensor()) {
          const auto& inputName = std::string(value->name());
          auto it = tensorValuesMeta.find(inputName);
          CHECK(it != tensorValuesMeta.end())
              << "Couldn't find " << inputName << " in tensorValuesMeta";
          validateInput(inputName, leaf.toTensor(), it->second);
        }
        executionFrame->setIValue(value->id(), leaf);
      }
      input_idx++;
    };
    leafApplyFromArgs(
        executionFrameFillUserInputs, args, kwargs, inputTreeSpec);
    outputs = graphExecutor_->executeWithPrefilledFrame(*executionFrame);
  } catch (...) {
    returnExecutorFrameToPool(std::move(executionFrame));
    throw;
  }

  returnExecutorFrameToPool(std::move(executionFrame));
  return *outputs;
}

ProfileMetrics Executor::benchmarkIndividualNodes(
    std::vector<std::vector<c10::IValue>> inputsList,
    const uint32_t warmupRuns,
    const uint32_t mainRuns) {
  CHECK(inputsList.size() > 0) << "Need at least one input to benchmark";
  CHECK(warmupRuns >= 1 && mainRuns >= 1) << "Need at least one run";

  for (const auto& inputs : inputsList) {
    for (const auto& input : inputs) {
      CHECK(input.isTensor() || input.isCustomClass())
          << "For now, all graph inputs should be tensor, or custom class object, but got "
          << input.tagKind();
    }
    if (executorConfig_.validateInputs) {
      validateInputs(inputs);
    }
  }
  auto executionFrame = getExecutorFrameFromPool();
  auto benchmarkResults = graphExecutor_->benchmarkIndividualNodes(
      *executionFrame, inputsList, warmupRuns, mainRuns);

  returnExecutorFrameToPool(std::move(executionFrame));
  return benchmarkResults;
}

std::vector<DelegateExecutor*> Executor::getDelegates() {
  std::vector<DelegateExecutor*> delegates;
  for (const auto& delegateExecutor : delegateExecutors_) {
    delegates.push_back(delegateExecutor.get());
  }
  return delegates;
}

} // namespace torch::nativert
