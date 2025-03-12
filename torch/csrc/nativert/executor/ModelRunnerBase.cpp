

#include "torch/csrc/nativert/executor/ModelRunnerBase.h"

#include "torch/csrc/nativert/common/RecordFunction.h"

#include <c10/util/Logging.h>
#include <caffe2/serialize/inline_container.h>

#include <fmt/format.h>

#include "torch/csrc/nativert/executor/Weights.h"
#include "torch/csrc/nativert/graph/TensorMeta.h"
#include "torch/csrc/nativert/package/pt2_archive_constants.h"

namespace torch::nativert {

ModelRunnerBase::ModelRunnerBase(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const std::string& modelName,
    ExecutorType executorType,
    const BaseRuntimeConfigs& runtimeConfigs,
    const std::function<Placement(const torch::nativert::Graph& graph)>&
        buildPlacementFn)
    : modelName_(modelName),
      executorType_(executorType),
      runtimeConfigs_(runtimeConfigs) {}

void ModelRunnerBase::setExecutorType(
    ExecutorType type,
    const std::string& platformArch) {
  LOG(INFO) << fmt::format(
      "Setting executor type to {} with platformArch='{}'", type, platformArch);
  executorType_ = type;
  if (type == ExecutorType::AOTINDUCTOR) {
    runtimeConfigs_.platformArch = platformArch;
  } else if (type == ExecutorType::MTIA) {
    // TODO: hardcoded for now (Sigmoid packages specify platformArch as
    // "mtia")
    runtimeConfigs_.platformArch = "mtia";
  }
}

const std::string& ModelRunnerBase::getModelName() const {
  return modelName_;
}

std::shared_ptr<Weights> ModelRunnerBase::getWeights() {
  if (executor_ != nullptr) {
    return executor_->getWeights();
  } else if (newWeights_ != nullptr) {
    return newWeights_;
  } else {
    TORCH_CHECK(
        false, "ModelRunner is not initialized, and no weights are loaded.");
  }
}

std::shared_ptr<Weights> ModelRunnerBase::loadNewWeights(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> packageStreamReader,
    std::function<bool(const std::string&)> skipSizeCheck,
    std::function<bool(const std::string&)> skipDtypeCheck) {
  LOG(INFO) << "ModelRunner loading new weights";
  newWeights_ = std::make_shared<Weights>(
      graph_.get(),
      packageStreamReader,
      stateDictPath_,
      archive_spec::WEIGHTS_DIR,
      constantPaths_,
      archive_spec::CONSTANTS_DIR,
      placement_,
      std::move(skipSizeCheck),
      std::move(skipDtypeCheck));

  return newWeights_;
}

void ModelRunnerBase::commitNewWeights() {
  CHECK(newWeights_) << "No new weights loaded";
  CHECK(executor_) << "ModelRunner not initialized";

  executor_->processWeights(newWeights_);

  executor_->atomicSwapWeights(std::move(newWeights_));

  newWeights_ = nullptr;
}

bool ModelRunnerBase::loadExtraFiles(
    ExtraFilesMap& extraFiles,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
        pytorchStreamReader) {
  auto filesExist = false;
  for (const auto& kv : extraFiles) {
    const auto key = std::string{archive_spec::EXTRA_DIR} + kv.first;
    if (pytorchStreamReader->hasRecord(key)) {
      auto [metaPtr, metaSize] = pytorchStreamReader->getRecord(key);
      extraFiles[kv.first] =
          std::string(static_cast<char*>(metaPtr.get()), metaSize);
      filesExist = true;
    }
  }
  return filesExist;
}

c10::IValue ModelRunnerBase::run(
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const RunConfigs& runConfigs) {
  return run({}, kwargs, runConfigs);
}

c10::IValue ModelRunnerBase::run(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const RunConfigs& runConfigs) {
  RecordFunction recordFunction("nativert::ModelRunner::run");

  CHECK(executor_) << "ModelRunner not initialized";

  // ModelRunner is only used for inference
  c10::InferenceMode mode;

  return treeUnflatten(
      executor_->execute(args, kwargs, inputSpec_), outputSpec_);
}

std::vector<c10::IValue> ModelRunnerBase::runWithFlatInputsAndOutputs(
    std::vector<c10::IValue>&& flatInputs,
    const RunConfigs& /* runConfigs */) {
  RecordFunction recordFunction(
      "nativert::ModelRunner::runWithFlatInputsAndOutputs");

  CHECK(executor_) << "ModelRunner not initialized";

  // ModelRunner is only used for inference
  c10::InferenceMode mode;

  return executor_->execute(std::move(flatInputs));
}

void ModelRunnerBase::benchmarkIndividualNodes(
    const std::vector<std::vector<c10::IValue>>& argsList,
    const std::vector<std::unordered_map<std::string, c10::IValue>>& kwargsList,
    const uint32_t warmupRuns,
    const uint32_t mainRuns,
    const bool printPerNodeTime,
    const RunConfigs& runConfigs) {
  std::vector<std::vector<c10::IValue>> flatInputsList;
  for (const auto& args : argsList) {
    if (!kwargsList.empty()) {
      for (const auto& kwargs : kwargsList) {
        flatInputsList.emplace_back(
            treeFlattenFromArgs(args, kwargs, inputSpec_));
      }
    } else {
      flatInputsList.emplace_back(treeFlattenFromArgs(args, {}, inputSpec_));
    }
  }
  c10::InferenceMode mode;
  auto results =
      executor_->benchmarkIndividualNodes(flatInputsList, warmupRuns, mainRuns);

  if (printPerNodeTime) {
    size_t i = 0;
    for (const auto& node : graph_->nodes()) {
      LOG(INFO) << "Node #" << i << ": " << node.toString()
                << "\n Time: " << results.timePerNode[i] << " ms/iter, ";
      i++;
    }
  }

  std::vector<std::pair<std::string, double>> sortedTimePerOp{
      results.timePerNodeType.begin(), results.timePerNodeType.end()};
  if (argsList.empty()) {
    // alphabetical sort
    std::sort(
        sortedTimePerOp.begin(),
        sortedTimePerOp.end(),
        [&results](auto& left, auto& right) {
          return results.instancesPerNodeType[left.first] >
              results.instancesPerNodeType[right.first];
        });
  } else {
    // sort by time
    std::sort(
        sortedTimePerOp.begin(),
        sortedTimePerOp.end(),
        [](auto& left, auto& right) { return left.second > right.second; });
  }

  LOG(INFO) << "Time per node type:" << '\n';
  std::ostringstream unsupportedNodeKinds;
  for (const auto& p : sortedTimePerOp) {
    const std::string& kind = p.first;
    const double ms = p.second;

    std::ostringstream oss;
    oss << std::setw(15) << ms << " ms. " << std::setw(10)
        << results.percentPerNodeType[kind] << "%. " << kind << " ("
        << results.instancesPerNodeType[kind] << " nodes";
    if (results.primNodes.count(kind)) {
      oss << ", prim) \n";
    } else if (results.staticDispatchNodes.count(kind)) {
      oss << ", static dispatch) \n";
    } else {
      unsupportedNodeKinds << kind << ", ";
      oss << ")\n";
    }
    LOG(INFO) << oss.str();
  }
  LOG(INFO) << std::setw(15) << results.totalTime << " ms. in Total" << '\n';
  LOG(INFO) << "Number of nodes: " << graph_->nodes().size() << '\n';

  auto unsupportedCount = results.totalNodesCount -
      results.staticDispatchNodesCount - results.primNodesCount;
  LOG(INFO) << "Total number of static dispatch nodes/total number of nodes: "
            << results.staticDispatchNodesCount << "/"
            << results.totalNodesCount << " ("
            << 100.0 * static_cast<float>(results.staticDispatchNodesCount) /
          static_cast<float>(results.totalNodesCount)
            << "%)" << '\n';
  LOG(INFO) << "Total number of prim nodes/total number of nodes: "
            << results.primNodesCount << "/" << results.totalNodesCount << " ("
            << 100.0 * static_cast<float>(results.primNodesCount) /
          static_cast<float>(results.totalNodesCount)
            << "%)" << '\n';
  LOG(INFO)
      << "Total number of nodes not covered by static dispatch/total number of nodes: "
      << unsupportedCount << "/" << results.totalNodesCount << " ("
      << 100.0 * static_cast<float>(unsupportedCount) /
          static_cast<float>(results.totalNodesCount)
      << "%)" << "\n Uncovered node kinds: " << unsupportedNodeKinds.str()
      << '\n';
}

std::vector<std::optional<c10::Device>>
ModelRunnerBase::getUserInputTargetDevices() const {
  std::vector<std::optional<c10::Device>> devices;
  for (const auto& tensorMeta : graph_->userInputsMeta()) {
    c10::Device targetDevice = placement_.getMappedDevice(tensorMeta.device());
    devices.push_back(targetDevice);
  }
  return devices;
}

std::pair<
    std::vector<c10::IValue>,
    std::unordered_map<std::string, c10::IValue>>
ModelRunnerBase::loadSampleInputs(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const Placement& placement) {
  LOG(INFO) << "Loading sample inputs in ModelRunner for model " << modelName_;

  std::string sampleInputsPath =
      fmt::format(archive_spec::SAMPLE_INPUTS_FILENAME_FORMAT, modelName_);

  CHECK(pytorchStreamReader->hasRecord(sampleInputsPath))
      << sampleInputsPath << " is not found in package";
  size_t size = pytorchStreamReader->getRecordSize(sampleInputsPath);
  std::vector<char> buffers(size);
  size_t sizeRead =
      pytorchStreamReader->getRecord(sampleInputsPath, buffers.data(), size);
  CHECK(sizeRead == size);

  c10::IValue value = torch::jit::pickle_load(buffers);

  // Move userInputs on the target device
  std::vector<TensorMeta> userInputsMeta = graph_->userInputsMeta();
  size_t tensorInputId = 0;
  value = treeMap(
      [&](const c10::IValue& inputVal) -> c10::IValue {
        if (inputVal.isTensor()) {
          auto& tensorMeta = userInputsMeta[tensorInputId];

          c10::Device targetDevice =
              placement.getMappedDevice(tensorMeta.device());
          auto r = inputVal.toTensor().to(targetDevice);

          VLOG(1) << "input #" << tensorInputId << " has been placed on "
                  << targetDevice;
          tensorInputId++;
          return r;
        } else {
          return inputVal;
        }
      },
      value,
      inputSpec_);

  CHECK(value.isTuple());
  CHECK(value.toTupleRef().elements().size() == 2);
  const auto& argsVal = value.toTupleRef().elements().at(0);
  const auto& kwargsVal = value.toTupleRef().elements().at(1);
  CHECK(argsVal.isTuple());
  CHECK(kwargsVal.isGenericDict());

  std::vector<c10::IValue> args;
  for (const auto& arg : argsVal.toTupleRef().elements()) {
    args.push_back(arg);
  }
  std::unordered_map<std::string, c10::IValue> kwargs;
  for (const auto& entry : kwargsVal.toGenericDict()) {
    kwargs[entry.key().toStringRef()] = entry.value();
  }
  return {std::move(args), std::move(kwargs)};
}

const std::vector<std::string>& ModelRunnerBase::getArgumentNames() const {
  return graph_->signature().userInputsVec();
}

c10::ArrayRef<const Value*> ModelRunnerBase::getArguments() const {
  return graph_->userInputs();
}

const TreeSpec& ModelRunnerBase::getOutputSpec() const {
  return outputSpec_;
}

const TreeSpec& ModelRunnerBase::getInputSpec() const {
  return inputSpec_;
}

std::string ModelRunnerBase::loadSerializedModel(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader)
    const {
  std::string modelFilePath =
      fmt::format(archive_spec::MODELS_FILENAME_FORMAT, modelName_);
  LOG(INFO) << "Loading model from: " << modelFilePath;

  CHECK(pytorchStreamReader->hasRecord(modelFilePath))
      << modelFilePath << " not found in package";
  const auto& [modelData, modelSize] =
      pytorchStreamReader->getRecord(modelFilePath);
  const std::string modelSerialized{
      reinterpret_cast<char*>(modelData.get()), modelSize};
  return modelSerialized;
}

void ModelRunnerBase::initialize(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
        pytorchStreamReader) {
  if (executor_ != nullptr) {
    LOG(WARNING)
        << "ModelRunner already initialized, re-initialization is an no op.";
    return;
  }

  initializeExecutor(newWeights_, std::move(pytorchStreamReader));
  newWeights_ = nullptr;
}

void ModelRunnerBase::initializeExecutor(
    std::shared_ptr<Weights> weights,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
        pytorchStreamReader) {
  CHECK(executor_ == nullptr) << "ModelRunner already initialized";
  weights->validateAllWeightsLoaded();

  ExecutorConfig config;
  config.maxParallelOps = runtimeConfigs_.maxParallelOps;
  config.validateInputs = runtimeConfigs_.validateInputs;
  config.enableStaticCPUKernels = runtimeConfigs_.enableStaticCPUKernels;
  config.maxNumConcurrentThreads = runtimeConfigs_.maxNumConcurrentThreads;
  config.enableStaticMemoryPlanning =
      runtimeConfigs_.enableStaticMemoryPlanning;
  config.runConstFolding = runtimeConfigs_.enableRuntimeConstFolding;

  if (executorType_ == ExecutorType::INTERPRETER) {
    executor_ = std::make_unique<Executor>(
        config, graph_, weights, placement_, pytorchStreamReader);
  } else if (executorType_ == ExecutorType::AOTINDUCTOR) {
    delegateGraph_ = deserializeDelegateGraph();
    delegateGraph_->applyDevicePlacement(placement_);
    VLOG(1) << "Delegate graph: \n" << *delegateGraph_;

    executor_ = std::make_unique<Executor>(
        config, delegateGraph_, weights, placement_, pytorchStreamReader);
  } else if (executorType_ == ExecutorType::MTIA) {
    delegateGraph_ = deserializeDelegateGraph();
    delegateGraph_->applyDevicePlacement(placement_);
    VLOG(1) << "Delegate graph: \n" << *delegateGraph_;
    config.modelName = modelName_;
    executor_ = std::make_unique<Executor>(
        config, delegateGraph_, weights, placement_, pytorchStreamReader);
  }
}

} // namespace torch::nativert
