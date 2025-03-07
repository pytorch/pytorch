#include "torch/csrc/runtime/executor/AOTIDelegateExecutor.h"
#include "torch/csrc/runtime/common/RecordFunction.h"

#include <c10/util/Logging.h>

#include "torch/csrc/runtime/executor/Weights.h"
#include "torch/csrc/utils/generated_serialization_types.h" // @manual=//caffe2:torch-cpp-cpu

namespace torch::runtime {

namespace {
template <typename T>
std::optional<at::ScalarType> parsePrecision(
    const std::optional<T>& precision) {
  if (precision) {
    return static_cast<at::ScalarType>(*precision);
  }
  return std::nullopt;
}

} // namespace

AOTIDelegateExecutor::AOTIDelegateExecutor(
    const std::string& path,
    std::shared_ptr<Weights> weights,
    c10::Device device,
    const ExecutorConfig& executorConfig,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> packageReader) {
  std::string aotInductorModelFileName = path + "/aotinductor_pickle_data.json";

  LOG(INFO) << "Loading aotinductor model from archive path: "
            << aotInductorModelFileName;

  CHECK(packageReader) << "Package reader cannot be null for lowered modules";
  CHECK(packageReader->hasRecord(aotInductorModelFileName))
      << "Missing record " << aotInductorModelFileName;
  const auto& [aotInductorModelData, aotInductorModelSize] =
      packageReader->getRecord(aotInductorModelFileName);

  const std::string aotInductorModelSerialized{
      reinterpret_cast<char*>(aotInductorModelData.get()),
      aotInductorModelSize};

  LOG(INFO) << "Loaded aot_inductor_model: " << aotInductorModelSerialized;

  auto aotInductorModel =
      nlohmann::json::parse(aotInductorModelSerialized)
          .template get<torch::_export::AOTInductorModelPickleData>();

  std::string tmpDir = extractToTemporaryFolder(packageReader, path);
  LOG(INFO) << "Extracted aot_inductor model to: " << tmpDir;

  std::string modelName = aotInductorModel.get_library_basename();
  std::string modelPath = tmpDir + "/" + modelName;
  std::string externKernelNodesPath =
      tmpDir + "/" + modelName.substr(0, modelName.size() - 3) + ".json";

  LOG(INFO) << "Creating AOTInductorModelImpl with device " << device.str();

  aotInductorModelImpl_ =
      std::make_unique<torch::aot_inductor::AOTInductorModelImpl>(
          modelPath,
          tmpDir,
          aotInductorModel.get_input_names(),
          aotInductorModel.get_output_names(),
          parsePrecision(aotInductorModel.get_floating_point_input_dtype()),
          parsePrecision(aotInductorModel.get_floating_point_output_dtype()),
          externKernelNodesPath,
          device.str(),
          /*num_runtimes*/ executorConfig.maxNumConcurrentThreads);

  auto constantInfos = aotInductorModelImpl_->getConstantInfos();
  for (const auto& [name, constantInfo] : constantInfos) {
    if (weights->contains(constantInfo.originalFqn)) {
      weightsNameMap_[constantInfo.originalFqn] = name;
    } else {
      LOG(WARNING)
          << "AOTI's Constant " << constantInfo.originalFqn
          << " is not found in weights, it's likely a constant created by AOTI constant folding. "
          << "Valid weight FQNs are " << weights->toString();
    }
  }

  // AOTI's DelegateExecutor doesn't need to call processWeights or
  // commitWeights here because it's invoked from Executor's ctor already.
}

void AOTIDelegateExecutor::processWeights(std::shared_ptr<Weights> weights) {
  LOG(INFO) << "AOTIDelegateExecutor processing weights";
  std::unordered_map<std::string, torch::Tensor*> newWeights;
  for (const auto& [original_fqn, name] : weightsNameMap_) {
    newWeights.emplace(name, &weights->at(original_fqn));
  }

  aotInductorModelImpl_->updateInactiveConstantBuffer(std::move(newWeights));
  aotInductorModelImpl_->runConstantFolding(/*use_inactive*/ true);
}

void AOTIDelegateExecutor::commitWeights() {
  LOG(INFO) << "AOTIDelegateExecutor committing weights";
  aotInductorModelImpl_->swapConstantBuffers();
}

std::vector<at::Tensor> AOTIDelegateExecutor::run(
    std::vector<at::Tensor>& inputs) {
  RecordFunction func("runtime::AOTIDelegateExecutor::run");

  std::vector<at::Tensor> outputs = aotInductorModelImpl_->forward(inputs);
  return outputs;
}
} // namespace torch::runtime
