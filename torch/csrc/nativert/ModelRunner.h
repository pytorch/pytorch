#pragma once

#include "torch/csrc/nativert/executor/ModelRunnerBase.h"

#include "torch/csrc/utils/generated_serialization_types.h" // @manual=//caffe2:torch-cpp-cpu

namespace torch::nativert::core {
class TORCH_API ModelRunner : public ModelRunnerBase {
 public:
  ModelRunner(
      const std::string& packagePath,
      const std::string& modelName,
      ExecutorType executorType,
      const BaseRuntimeConfigs& runtimeConfigs,
      const Placement& placement = Placement());

  ModelRunner(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
      const std::string& modelName,
      ExecutorType executorType,
      const BaseRuntimeConfigs& runtimeConfigs,
      const Placement& placement = Placement());

  ModelRunner(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader,
      const std::string& modelName,
      ExecutorType executorType,
      const BaseRuntimeConfigs& runtimeConfigs,
      const Placement& placement = Placement());

  ModelRunner(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader,
      const std::string& modelName,
      ExecutorType executorType,
      const BaseRuntimeConfigs& runtimeConfigs,
      // functor to build the placement after the graph is loaded, but before
      // loading the weights.
      const std::function<Placement(const torch::nativert::Graph& graph)>&
          buildPlacementFn);

  ModelRunner(ModelRunner&&) = default;
  ModelRunner& operator=(ModelRunner&&) = default;
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ~ModelRunner() override = default;

  std::vector<std::string> availableDelegates() const {
    std::vector<std::string> delegateNames;
    delegateNames.reserve(delegates_.size());
    for (const auto& [name, _] : delegates_) {
      delegateNames.push_back(name);
    }
    return delegateNames;
  }

  template <typename T>
  std::vector<T*> getDelegates() {
    std::vector<T*> delegates;
    for (const auto& delegate : executor_->getDelegates()) {
      if (auto* d = dynamic_cast<T*>(delegate)) {
        delegates.push_back(d);
      }
    }
    return delegates;
  }

 private:
  std::unique_ptr<Graph> deserializeDelegateGraph() const override;

  torch::_export::Model model_;
  torch::_export::ExportedProgram exportedProgram_;
  std::unordered_map<std::string, torch::_export::ExportedProgram> delegates_;
};
} // namespace torch::nativert::core
