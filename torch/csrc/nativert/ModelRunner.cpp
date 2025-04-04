#include "torch/csrc/nativert/ModelRunner.h"

#include <nlohmann/json.hpp>

#include <caffe2/serialize/file_adapter.h>

#include "torch/csrc/nativert/graph/GraphPasses.h"
#include "torch/csrc/nativert/graph/Serialization.h"

namespace torch::nativert::core {

ModelRunner::ModelRunner(
    const std::string& packagePath,
    const std::string& modelName,
    ExecutorType executorType,
    const BaseRuntimeConfigs& runtimeConfigs,
    const Placement& placement)
    : ModelRunner(
          std::make_unique<caffe2::serialize::FileAdapter>(packagePath),
          modelName,
          executorType,
          runtimeConfigs,
          placement) {}

ModelRunner::ModelRunner(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    const std::string& modelName,
    ExecutorType executorType,
    const BaseRuntimeConfigs& runtimeConfigs,
    const Placement& placement)
    : ModelRunner(
          std::make_shared<caffe2::serialize::PyTorchStreamReader>(rai),
          modelName,
          executorType,
          runtimeConfigs,
          placement) {}

ModelRunner::ModelRunner(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const std::string& modelName,
    ExecutorType executorType,
    const BaseRuntimeConfigs& runtimeConfigs,
    const Placement& placement)
    : ModelRunner(
          std::move(pytorchStreamReader),
          modelName,
          executorType,
          runtimeConfigs,
          [=](const torch::nativert::Graph&) { return placement; }) {}

ModelRunner::ModelRunner(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const std::string& modelName,
    ExecutorType executorType,
    const BaseRuntimeConfigs& runtimeConfigs,
    const std::function<Placement(const torch::nativert::Graph& graph)>&
        buildPlacementFn)
    : ModelRunnerBase(
          pytorchStreamReader,
          modelName,
          executorType,
          runtimeConfigs,
          buildPlacementFn) {
  std::string modelSerialized = loadSerializedModel(pytorchStreamReader);

  model_ = nlohmann::json::parse(modelSerialized)
               .template get<torch::_export::Model>();
  exportedProgram_ = model_.get_program().get_methods().at("forward");
  for (const auto& _ : model_.get_delegates()) {
    (void)_;
    TORCH_CHECK(false, "Delegates are not supported yet");
    // TODO delegates_.emplace(name, delegate.get_methods().at("forward"));
  }
  stateDictPath_ = model_.get_tensorPaths();
  constantPaths_ = model_.get_constantPaths();
  TORCH_CHECK_EQ(
      exportedProgram_.get_graph_module().get_module_call_graph()[0].get_fqn(),
      "");

  inputSpec_ = treeSpecLoads(exportedProgram_.get_graph_module()
                                 .get_module_call_graph()[0]
                                 .get_signature()
                                 .value()
                                 .get_in_spec());
  outputSpec_ = treeSpecLoads(exportedProgram_.get_graph_module()
                                  .get_module_call_graph()[0]
                                  .get_signature()
                                  .value()
                                  .get_out_spec());

  graph_ = jsonToGraph(
      model_.get_program().get_methods().at("forward").get_graph_module());

  VLOG(1) << "Graph: \n" << *graph_;

  placement_ = buildPlacementFn(*graph_);
  LOG(INFO) << "Placement: " << placement_;

  graph_->applyDevicePlacement(placement_);
  selectScalarOverload(graph_.get());

  loadNewWeights(pytorchStreamReader);

  if (!runtimeConfigs.deferInitialization) {
    initialize(pytorchStreamReader);
  }
}

std::unique_ptr<Graph> ModelRunner::deserializeDelegateGraph() const {
  return {}; // TODO
}

} // namespace torch::nativert::core
