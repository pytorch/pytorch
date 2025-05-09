#pragma once

#include <string>
#include <unordered_set>

#include <c10/util/Logging.h>

#include <torch/csrc/utils/generated_serialization_types.h>

namespace torch::nativert {

/**
 * @brief An in-memory representation for input and output specs of a graph.
 *
 * The GraphSignature class models the input and output specs of an exported
 * graph produced by torch.export, which is a fx.Graph with stronger invariants
 * gurantees. It holds the graph information deserialized from the pt2 archive
 * package. Runtime relies on the GraphSignature for weight name lookup and
 * weight loading. The serialization schema is defined in
 * torch/_export/serde/schema.py See more at:
 * https://docs.pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature
 */
class GraphSignature {
 public:
  GraphSignature() = default;
  explicit GraphSignature(const torch::_export::GraphSignature& storage);

  const auto& lossOutput() const {
    return lossOutput_;
  }

  const auto& gradientsToParameters() const {
    return gradientsToParameters_;
  }

  const auto& gradientsToUserInputs() const {
    return gradientsToUserInputs_;
  }

  const auto& inputsToParameters() const {
    return inputsToParameters_;
  }

  const auto& inputsToBuffers() const {
    return inputsToBuffers_;
  }

  const auto& inputsToTensorConstants() const {
    return inputsToTensorConstants_;
  }

  const auto& inputsToCustomObjs() const {
    return inputsToCustomObjs_;
  }

  const auto& parameters() const {
    return parameters_;
  }

  const auto& buffers() const {
    return buffers_;
  }

  const auto& nonPersistentBuffers() const {
    return nonPersistentBuffers_;
  }

  const auto& tensorConstants() const {
    return tensorConstants_;
  }

  const auto& customObjs() const {
    return customObjs_;
  }

  const auto& userInputs() const {
    return userInputs_;
  }

  const auto& constantInputs() const {
    return constantInputs_;
  }

  const auto& userOutputs() const {
    return userOutputs_;
  }

  const auto& buffersToMutate() const {
    return buffersToMutate_;
  }

  const auto& userInputsToMutate() const {
    return userInputsToMutate_;
  }

  bool hasBackward() const {
    return !(
        lossOutput_.empty() && gradientsToParameters_.empty() &&
        gradientsToUserInputs_.empty() && buffersToMutate_.empty());
  }

  // Mapping of FQNs to weights with stable iteration order.
  const auto& inputsToWeights() const {
    return inputsToWeights_;
  }

  void lint(
      const std::unordered_set<std::string>& graphInputs,
      const std::unordered_set<std::string>& graphOutputs) const;
  void replaceAllUses(std::string_view old, std::string_view replacement);

  torch::_export::GraphSignature serialize() const;

 private:
  std::unordered_set<std::string> inputNames() const;
  std::unordered_set<std::optional<std::string>> outputNames() const;

  std::unordered_map<std::string, std::string> gradientsToParameters_;
  std::unordered_map<std::string, std::string> gradientsToUserInputs_;
  std::unordered_map<std::string, std::string> inputsToParameters_;
  std::unordered_map<std::string, std::string> inputsToBuffers_;
  std::unordered_map<std::string, std::string> inputsToTensorConstants_;
  std::unordered_map<std::string, std::string> inputsToCustomObjs_;
  std::unordered_map<std::string, std::string> buffersToMutate_;
  std::unordered_map<std::string, std::string> userInputsToMutate_;

  // map union of inputsToParameters_, inputsToBuffers_ and
  // inputsToTensorConstants_
  std::vector<std::pair<std::string, std::string>> inputsToWeights_;

  std::vector<std::string> parameters_;
  std::vector<std::string> buffers_;
  std::vector<std::string> tensorConstants_;
  std::vector<std::string> customObjs_;
  std::vector<std::string> nonPersistentBuffers_;

  std::vector<std::string> userInputs_;
  std::vector<std::string> constantInputs_;
  std::vector<std::optional<std::string>> userOutputs_;
  std::string lossOutput_;
};

std::ostream& operator<<(std::ostream& out, const GraphSignature& sig);

} // namespace torch::nativert
