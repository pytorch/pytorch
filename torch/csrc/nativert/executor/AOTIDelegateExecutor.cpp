#include "torch/csrc/nativert/executor/AOTIDelegateExecutor.h"
#include "torch/csrc/nativert/common/RecordFunction.h"

#include <c10/util/Logging.h>

#include "torch/csrc/nativert/executor/Weights.h"
#include "torch/csrc/nativert/package/pt2_archive_constants.h"

#include "torch/csrc/utils/generated_serialization_types.h" // @manual=//caffe2:torch-cpp-cpu

namespace torch::nativert {

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

  // We have to read the custom_objs_config.json,
  // because the weights->customObjs_ keys are not the same as the arg names
  // in the externKernelNodesPath.
  std::string customObjsJsonPath = tmpDir + "/custom_objs_config.json";

  std::ifstream customObjsJsonFile(customObjsJsonPath);
  std::unordered_map<std::string, c10::IValue> custom_objs;
  if (!customObjsJsonFile.is_open()) {
    // BC-compatible with old files that don't have custom_objs_config.json
    LOG(INFO) << "Unable to open file " + customObjsJsonPath;
  } else {
    LOG(INFO) << "Load custom object mapping from: " << customObjsJsonPath;

    nlohmann::json customObjsJson;
    customObjsJsonFile >> customObjsJson;

    // Populate custom_objs with the custom object names from the json file,
    // and the c10::IValue from the weights.
    for (auto& [customObjName, file_name] : customObjsJson.items()) {
      custom_objs[customObjName] = weights->getCustomObjByFileName(
          std::string(archive_spec::CONSTANTS_DIR) +
          file_name.get<std::string>());
      LOG(INFO) << "Copy custom object to FbProxyExecutor: " << customObjName
                << " from " << file_name;
    }
  }
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
          /*num_runtimes*/ executorConfig.maxNumConcurrentThreads,
          /*custom_objs*/ std::move(custom_objs));

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
  RecordFunction func("nativert::AOTIDelegateExecutor::run");

  std::vector<at::Tensor> outputs = aotInductorModelImpl_->forward(inputs);
  return outputs;
}
} // namespace torch::nativert
