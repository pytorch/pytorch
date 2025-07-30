#pragma once

#include <fmt/format.h>

#include <c10/macros/Export.h>
#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/detail/ITree.h>
#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/executor/Placement.h>

namespace torch::nativert {
class TORCH_API ModelRunner {
 public:
  ModelRunner(const std::string& packagePath, const std::string& modelName);

  ModelRunner(ModelRunner&&) = default;
  ModelRunner& operator=(ModelRunner&&) = default;
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ~ModelRunner() = default;

  c10::IValue run(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  /**
   * A low level API which expects user to always pass in flattened inputs.
   * The ownership of the entire input list must be transferred to the
   * executor via std::move or in-place construction.
   */
  std::vector<c10::IValue> runWithFlatInputsAndOutputs(
      std::vector<c10::IValue> flatInputs);

 private:
  // original non-delegated graph from torch.export()
  std::shared_ptr<Graph> graph_;

  std::unique_ptr<Executor> executor_;

  ITreeSpec inputSpec_;
  ITreeSpec outputSpec_;

  torch::_export::ExportedProgram exportedProgram_;
};
} // namespace torch::nativert
