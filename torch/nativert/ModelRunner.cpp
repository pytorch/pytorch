#include <torch/nativert/ModelRunner.h>

#include <variant>

#include <nlohmann/json.hpp>

#include <caffe2/serialize/file_adapter.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/graph/GraphPasses.h>
#include <torch/nativert/graph/Serialization.h>

namespace torch::nativert {

using torch::nativert::jsonToGraph;
using torch::nativert::detail::itreeSpecLoads;

namespace {
std::shared_ptr<Weights> loadWeightsDefault(
    Graph& graph,
    caffe2::serialize::PyTorchStreamReader& reader,
    std::string_view modelName) {
  auto weightsPath = fmt::format(
      "{}{}.pt", torch::_export::archive_spec::WEIGHTS_DIR, modelName);
  auto constantsPath = fmt::format(
      "{}{}.pt", torch::_export::archive_spec::CONSTANTS_DIR, modelName);
  TORCH_CHECK(
      reader.hasRecord(weightsPath), weightsPath, " not found in package");
  TORCH_CHECK(
      reader.hasRecord(constantsPath), constantsPath, " not found in package");
  const auto& [weightsData, weightsSize] = reader.getRecord(weightsPath);
  auto weights =
      torch::jit::pickle_load_obj(
          std::string_view{static_cast<char*>(weightsData.get()), weightsSize})
          .toGenericDict();
  const auto& [constantsData, constantsSize] = reader.getRecord(constantsPath);
  auto constants =
      torch::jit::pickle_load_obj(
          std::string_view{
              static_cast<char*>(constantsData.get()), constantsSize})
          .toGenericDict();
  std::unordered_map<std::string, c10::IValue> stateDict;
  std::unordered_map<std::string, c10::IValue> constantsDict;
  for (const auto& item : weights) {
    stateDict[item.key().toStringRef()] = item.value();
  }
  for (const auto& item : constants) {
    constantsDict[item.key().toStringRef()] = item.value();
  }
  return std::make_shared<Weights>(&graph, stateDict, constantsDict);
}
} // namespace

ModelRunner::ModelRunner(
    const std::string& packagePath,
    const std::string& modelName) {
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

  auto weights = loadWeightsDefault(*graph_, *pytorchStreamReader, modelName);

  weights->validateAllWeightsLoaded();

  torch::nativert::ExecutorConfig config;

  executor_ = std::make_unique<Executor>(
      config, graph_, std::move(weights), pytorchStreamReader);
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

} // namespace torch::nativert
