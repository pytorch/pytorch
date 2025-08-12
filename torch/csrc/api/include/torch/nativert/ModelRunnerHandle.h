#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>

namespace torch::nativert {

// We don't want to forward declare in general but including ModelRunner will
// pollute the public API namespace too much. Therefore, we just use pimpl an
// incomplete ModelRunner here.
class ModelRunner;

class TORCH_API ModelRunnerHandle {
 public:
  ModelRunnerHandle(
      const std::string& packagePath,
      const std::string& modelName);

  ModelRunnerHandle(ModelRunnerHandle&&) = default;
  ModelRunnerHandle& operator=(ModelRunnerHandle&&) = default;
  ModelRunnerHandle(const ModelRunnerHandle&) = delete;
  ModelRunnerHandle& operator=(const ModelRunnerHandle&) = delete;
  ~ModelRunnerHandle();

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
  std::unique_ptr<ModelRunner> impl_;
};

} // namespace torch::nativert
