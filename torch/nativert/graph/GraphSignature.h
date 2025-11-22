#pragma once

#include <string>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>

#include <torch/csrc/utils/generated_serialization_types.h>

namespace torch::nativert {

/**
 * @brief An in-memory representation for input and output specs of a graph.
 *
 * The GraphSignature class models the input and output specs of an exported
 * graph produced by torch.export, which is a fx.Graph with stronger invariants
 * guarantees. It holds the graph information deserialized from the pt2 archive
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

  auto inputsToParameters() const {
    c10::FastMap<std::string_view, std::string_view> inputsToParameters;
    inputsToParameters.reserve(numParameters_);
    for (int i = 0; i < numParameters_; ++i) {
      inputsToParameters.emplace(
          inputsToWeights_[i].first, inputsToWeights_[i].second);
    }
    return inputsToParameters;
  }

  auto inputsToBuffers() const {
    c10::FastMap<std::string_view, std::string_view> inputsToBuffers;
    inputsToBuffers.reserve(numPersistentBuffers_ + numNonPersistentBuffers_);
    for (int i = numParameters_;
         i < numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_;
         ++i) {
      inputsToBuffers.emplace(
          inputsToWeights_[i].first, inputsToWeights_[i].second);
    }
    return inputsToBuffers;
  }

  auto inputsToTensorConstants() const {
    c10::FastMap<std::string_view, std::string_view> inputsToTensorConstants;
    inputsToTensorConstants.reserve(numTensorConstants_);
    for (int i =
             numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_;
         i < numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_ +
             numTensorConstants_;
         ++i) {
      inputsToTensorConstants.emplace(
          inputsToWeights_[i].first, inputsToWeights_[i].second);
    }
    return inputsToTensorConstants;
  }

  const auto& inputsToCustomObjs() const {
    return inputsToCustomObjs_;
  }

  auto parameters() const {
    std::vector<std::string_view> parameters;
    parameters.reserve(numParameters_);
    for (int i = 0; i < numParameters_; ++i) {
      parameters.emplace_back(inputsToWeights_[i].second);
    }
    return parameters;
  }

  auto buffers() const {
    std::vector<std::string_view> buffers;
    buffers.reserve(numPersistentBuffers_);
    for (int i = numParameters_; i < numParameters_ + numPersistentBuffers_;
         i++) {
      buffers.emplace_back(inputsToWeights_[i].second);
    }
    return buffers;
  }

  auto nonPersistentBuffers() const {
    std::vector<std::string_view> buffers;
    buffers.reserve(numNonPersistentBuffers_);
    for (int i = numParameters_ + numPersistentBuffers_;
         i < numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_;
         i++) {
      buffers.emplace_back(inputsToWeights_[i].second);
    }
    return buffers;
  }

  auto tensorConstants() const {
    std::vector<std::string_view> tensorConstants;
    tensorConstants.reserve(numTensorConstants_);
    for (int i =
             numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_;
         i < numParameters_ + numPersistentBuffers_ + numNonPersistentBuffers_ +
             numTensorConstants_;
         i++) {
      tensorConstants.emplace_back(inputsToWeights_[i].second);
    }
    return tensorConstants;
  }

  auto customObjs() const {
    std::vector<std::string_view> customObjs;
    customObjs.reserve(numCustomObjs_);
    for (int i = 0; i < numCustomObjs_; ++i) {
      customObjs.emplace_back(inputsToCustomObjs_[i].second);
    }
    return customObjs;
  }

  const auto& userInputs() const {
    return userInputs_;
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
      const c10::FastSet<std::string>& graphInputs,
      const c10::FastSet<std::string>& graphOutputs) const;
  void replaceAllUses(std::string_view old, std::string_view replacement);

  torch::_export::GraphSignature serialize() const;

 private:
  c10::FastSet<std::string> inputNames() const;
  c10::FastSet<std::optional<std::string>> outputNames() const;

  c10::FastMap<std::string, std::string> gradientsToParameters_;
  c10::FastMap<std::string, std::string> gradientsToUserInputs_;
  c10::FastMap<std::string, std::string> buffersToMutate_;
  c10::FastMap<std::string, std::string> userInputsToMutate_;

  // Order is [inputsToParameters, inputsToBuffers,
  // inputsToNonPersistentBuffers, inputsToTensorConstants]
  // We need to maintain the order of these weight names as it is
  // an important assumption in nativert for weight loading and
  // unused weight optimization in Weights.cpp
  std::vector<std::pair<std::string, std::string>> inputsToWeights_;
  int numParameters_ = 0;
  int numPersistentBuffers_ = 0;
  int numNonPersistentBuffers_ = 0;
  int numTensorConstants_ = 0;
  int numCustomObjs_ = 0;

  std::vector<std::pair<std::string, std::string>> inputsToCustomObjs_;

  std::vector<std::string> userInputs_;
  std::vector<std::optional<std::string>> userOutputs_;
  std::string lossOutput_;
};

std::ostream& operator<<(std::ostream& out, const GraphSignature& sig);

} // namespace torch::nativert
