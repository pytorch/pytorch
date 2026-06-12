#include <torch/nativert/ModelRunner.h>

#include <variant>

#include <nlohmann/json.hpp>

#include <caffe2/serialize/file_adapter.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/graph/GraphPasses.h>
#include <torch/nativert/graph/Serialization.h>
#include <torch/nativert/kernels/KernelHandlerRegistry.h>

namespace torch::nativert {

using torch::nativert::jsonToGraph;
using torch::nativert::detail::itreeSpecLoads;

ModelRunner::ModelRunner(
    const std::string& packagePath,
    const std::string& modelName) {
  register_kernel_handlers();
  auto pytorchStreamReader =
      std::make_shared<caffe2::serialize::PyTorchStreamReader>(
          std::make_unique<caffe2::serialize::FileAdapter>(packagePath));
  std::string modelFilePath = fmt::format(
      torch::_export::archive_spec::MODELS_FILENAME_FORMAT, modelName);
  LOG(INFO) << "Loading model from: " << modelFilePath;

  TORCH_CHECK(
      pytorchStreamReader->hasRecord(modelFilePath),
      modelFilePath,
      " not found in package");
  const auto& [modelData, modelSize] =
      pytorchStreamReader->getRecord(modelFilePath);
  const std::string modelSerialized{
      reinterpret_cast<char*>(modelData.get()), modelSize};

  exportedProgram_ = nlohmann::json::parse(modelSerialized)
                         .template get<torch::_export::ExportedProgram>();

  TORCH_CHECK(exportedProgram_.get_graph_module()
                  .get_module_call_graph()[0]
                  .get_fqn()
                  .empty());

  tensorPaths_ = getPayloadConfig(
      pytorchStreamReader,
      torch::_export::archive_spec::WEIGHTS_CONFIG_FILENAME_FORMAT,
      modelName);

  constantPaths_ = getPayloadConfig(
      pytorchStreamReader,
      torch::_export::archive_spec::CONSTANTS_CONFIG_FILENAME_FORMAT,
      modelName);

  graph_ = jsonToGraph(exportedProgram_.get_graph_module());

  std::vector<const Value*> userInputs(
      graph_->userInputs().begin(), graph_->userInputs().end());
  const auto& signatureOpt = exportedProgram_.get_graph_module()
                                 .get_module_call_graph()[0]
                                 .get_signature();
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& signature = signatureOpt.value();
  inputSpec_ = itreeSpecLoads(signature.get_in_spec(), userInputs);

  const auto& userOutputs = graph_->userOutputs();
  std::vector<const Value*> updatedUserOutput(userOutputs.size(), nullptr);
  for (size_t i = 0; i < userOutputs.size(); ++i) {
    if (const auto* valuePtr = std::get_if<Value*>(&userOutputs[i])) {
      updatedUserOutput[i] = *valuePtr;
    }
  }
  outputSpec_ = itreeSpecLoads(signature.get_out_spec(), updatedUserOutput);

  torch::nativert::Placement placement;

  graph_->applyDevicePlacement(placement);
  selectScalarOverload(graph_.get());

  auto weights = loadWeightsDefault(*graph_, pytorchStreamReader);

  weights->validateAllWeightsLoaded();

  torch::nativert::ExecutorConfig config;
  config.modelName = modelName;

  executor_ = std::make_unique<Executor>(
      config, graph_, std::move(weights), pytorchStreamReader);
}

std::unordered_map<std::string, std::string> ModelRunner::getPayloadConfig(
    const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
        pytorchStreamReader,
    std::string_view configFilenameFormat,
    const std::string& modelName) {
  std::string configPath =
      fmt::format(fmt::runtime(configFilenameFormat), modelName);

  TORCH_CHECK(
      pytorchStreamReader->hasRecord(configPath),
      configPath,
      " not found in package");

  const auto& [configData, configSize] =
      pytorchStreamReader->getRecord(configPath);
  const std::string configSerialized{
      reinterpret_cast<char*>(configData.get()), configSize};

  auto configJson = nlohmann::json::parse(configSerialized)
                        .template get<torch::_export::PayloadConfig>();
  auto config = configJson.get_config();
  std::unordered_map<std::string, std::string> targetPaths;
  for (const auto& configEntry : config) {
    targetPaths[configEntry.first] = configEntry.second.get_path_name();
  }
  return targetPaths;
}

std::shared_ptr<Weights> ModelRunner::loadWeightsDefault(
    Graph& graph,
    const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>& reader) {
  return std::make_shared<Weights>(
      &graph,
      reader,
      tensorPaths_,
      torch::_export::archive_spec::WEIGHTS_DIR,
      constantPaths_,
      torch::_export::archive_spec::CONSTANTS_DIR);
}

c10::IValue ModelRunner::run(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  TORCH_CHECK(executor_, "ModelRunner not initialized");

  // ModelRunner is only used for inference
  c10::InferenceMode mode;

  return itreeUnflatten(
      executor_->execute(args, kwargs, inputSpec_), outputSpec_);
}

std::vector<c10::IValue> ModelRunner::runWithFlatInputsAndOutputs(
    std::vector<c10::IValue> flatInputs) {
  TORCH_CHECK(executor_, "ModelRunner not initialized");

  // ModelRunner is only used for inference
  c10::InferenceMode mode;

  return executor_->execute(std::move(flatInputs));
}

uint64_t ModelRunner::numOutputs() const {
  TORCH_CHECK(executor_, "ModelRunner not initialized");
  return executor_->graphSignature().userOutputs().size();
}

ModelRunnerHandle::ModelRunnerHandle(
    const std::string& packagePath,
    const std::string& modelName)
    : impl_(std::make_unique<ModelRunner>(packagePath, modelName)) {}
ModelRunnerHandle::~ModelRunnerHandle() = default;

c10::IValue ModelRunnerHandle::run(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return impl_->run(args, kwargs);
}

std::vector<c10::IValue> ModelRunnerHandle::runWithFlatInputsAndOutputs(
    std::vector<c10::IValue> flatInputs) {
  return impl_->runWithFlatInputsAndOutputs(std::move(flatInputs));
}

} // namespace torch::nativert
