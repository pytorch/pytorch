#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

#include "c10/util/Exception.h"
#include "torch/csrc/nativert/graph/GraphSignature.h"
#include "torch/csrc/utils/generated_serialization_types.h" // @manual=//caffe2:torch-cpp-cpu

namespace torch::nativert {

namespace {

bool isSymbolicOutput(torch::_export::Argument::Tag t) {
  switch (t) {
    case torch::_export::Argument::Tag::AS_TENSOR:
    case torch::_export::Argument::Tag::AS_TENSORS:
    case torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS:
    case torch::_export::Argument::Tag::AS_SYM_BOOL:
    case torch::_export::Argument::Tag::AS_SYM_BOOLS:
    case torch::_export::Argument::Tag::AS_SYM_INT:
    case torch::_export::Argument::Tag::AS_SYM_INTS:
    case torch::_export::Argument::Tag::AS_SYM_FLOAT:
    case torch::_export::Argument::Tag::AS_SYM_FLOATS:
    case torch::_export::Argument::Tag::AS_CUSTOM_OBJ:
      return true;
    default:
      return false;
  }
}

std::pair<std::string, std::string> getSpecDetails(
    const torch::_export::InputSpec& inputSpec) {
  // Retrieve the argument name and spec tag name
  std::string argName;
  std::string tagName;
  switch (inputSpec.tag()) {
    case torch::_export::InputSpec::Tag::PARAMETER:
      argName = inputSpec.get_parameter().get_arg().get_name();
      tagName = "PARAMETER";
      break;
    case torch::_export::InputSpec::Tag::BUFFER:
      argName = inputSpec.get_buffer().get_arg().get_name();
      tagName = "BUFFER";
      break;
    case torch::_export::InputSpec::Tag::TENSOR_CONSTANT:
      argName = inputSpec.get_tensor_constant().get_arg().get_name();
      tagName = "TENSOR_CONSTANT";
      break;
    case torch::_export::InputSpec::Tag::CUSTOM_OBJ:
      tagName = "CUSTOM_OBJ";
      argName = inputSpec.get_custom_obj().get_arg().get_name();
      break;
    case torch::_export::InputSpec::Tag::USER_INPUT:
      tagName = "USER_INPUT";
      if (inputSpec.get_user_input().get_arg().tag() ==
          torch::_export::Argument::Tag::AS_TENSOR) {
        argName =
            inputSpec.get_user_input().get_arg().get_as_tensor().get_name();
      } else if (
          inputSpec.get_user_input().get_arg().tag() ==
          torch::_export::Argument::Tag::AS_CUSTOM_OBJ) {
        argName =
            inputSpec.get_user_input().get_arg().get_as_custom_obj().get_name();
      } else {
        throw std::runtime_error("Unsupported USER_INPUT argument type.");
      }
      break;
    case torch::_export::InputSpec::Tag::CONSTANT_INPUT:
      argName = inputSpec.get_constant_input().get_name();
      tagName = "CONSTANT_INPUT";
      break;
    case torch::_export::InputSpec::Tag::TOKEN:
      throw std::runtime_error("Token inputs not implemented yet");
    default:
      throw std::runtime_error("Unknown InputSpec tag encountered.");
  }
  return std::make_pair(argName, tagName);
}

void checkInputOrders(
    const std::vector<torch::_export::InputSpec>& inputSpecs) {
  // Map each tag to its index in the expected order
  std::unordered_map<torch::_export::InputSpec::Tag, size_t> tagOrderMap = {
      {torch::_export::InputSpec::Tag::TOKEN, 0},
      {torch::_export::InputSpec::Tag::PARAMETER, 1},
      {torch::_export::InputSpec::Tag::BUFFER, 2},
      {torch::_export::InputSpec::Tag::TENSOR_CONSTANT, 3},
      {torch::_export::InputSpec::Tag::CUSTOM_OBJ, 4}};
  size_t currentOrderIndex = 0;
  bool seenNonPersistentBuffer = false;
  for (const auto& inputSpec : inputSpecs) {
    if (inputSpec.tag() == torch::_export::InputSpec::Tag::USER_INPUT ||
        inputSpec.tag() == torch::_export::InputSpec::Tag::CONSTANT_INPUT) {
      continue;
    }
    auto it = tagOrderMap.find(inputSpec.tag());
    if (it == tagOrderMap.end()) {
      throw std::runtime_error("Unknown InputSpec tag encountered.");
    }
    size_t tagIndex = it->second;

    if (tagIndex < currentOrderIndex) {
      auto [argName, tagName] = getSpecDetails(inputSpec);
      throw std::runtime_error(fmt::format(
          "Input arg {} with InputSpec {} is out of order!", argName, tagName));
    }
    currentOrderIndex = tagIndex;
    // Additional check for buffers
    if (inputSpec.tag() == torch::_export::InputSpec::Tag::BUFFER) {
      if (!inputSpec.get_buffer().get_persistent()) {
        seenNonPersistentBuffer = true;
      } else if (
          inputSpec.get_buffer().get_persistent() && seenNonPersistentBuffer) {
        throw std::runtime_error(
            "Persistent buffer found after a non-persistent buffer. "
            "Persistent buffers must come before non-persistent buffers.");
      }
    }
  }
}

void checkInputNames(
    const std::set<std::string>& sigNames,
    const std::set<std::string>& graphNames) {
  if (sigNames == graphNames) {
    return;
  }

  std::string errorMsg =
      "Error: Value name difference detected between graph signature and graph nodes:\n";
  errorMsg += "Signature value names:\n";
  errorMsg += fmt::format("[{}]\n", fmt::join(sigNames, ", "));
  errorMsg += "Graph node names:\n";
  errorMsg += fmt::format("[{}]", fmt::join(graphNames, ", "));
  LOG(FATAL) << errorMsg;
};

void checkOutputNames(
    const std::set<std::optional<std::string>>& sigNames,
    const std::set<std::string>& graphNames) {
  std::vector<std::string> validNames;
  for (const auto& nameOpt : sigNames) {
    if (nameOpt.has_value()) {
      validNames.push_back(*nameOpt);
    }
  }

  for (const auto& name : validNames) {
    if (graphNames.find(name) == graphNames.end()) {
      std::string errorMsg =
          "Error: Value name difference detected between graph signature and graph nodes:\n";
      errorMsg += "Signature value names:\n";
      errorMsg += fmt::format("[{}]\n", fmt::join(validNames, ", "));
      errorMsg += "Graph node names:\n";
      errorMsg += fmt::format("[{}]", fmt::join(graphNames, ", "));
      LOG(FATAL) << errorMsg;
    }
  }
};

void replaceInMap(
    std::unordered_map<std::string, std::string>& map,
    std::string_view old,
    std::string_view replacement) {
  auto it = map.find(std::string{old});
  if (it == map.end()) {
    return;
  }
  std::string value = it->second;
  map.erase(it);
  map.emplace(replacement, value);
}

} // namespace

GraphSignature::GraphSignature(const torch::_export::GraphSignature& storage) {
  checkInputOrders(storage.get_input_specs());

  for (const torch::_export::InputSpec& inputSpec : storage.get_input_specs()) {
    switch (inputSpec.tag()) {
      case torch::_export::InputSpec::Tag::USER_INPUT: {
        if (inputSpec.get_user_input().get_arg().tag() ==
            torch::_export::Argument::Tag::AS_TENSOR) {
          userInputs_.push_back(
              inputSpec.get_user_input().get_arg().get_as_tensor().get_name());
        } else if (
            inputSpec.get_user_input().get_arg().tag() ==
            torch::_export::Argument::Tag::AS_CUSTOM_OBJ) {
          userInputs_.push_back(inputSpec.get_user_input()
                                    .get_arg()
                                    .get_as_custom_obj()
                                    .get_name());
        } else {
          // TODO: handle other types
          LOG(FATAL) << "Non tensor inputs nyi";
        }
        break;
      }
      case torch::_export::InputSpec::Tag::PARAMETER: {
        parameters_.push_back(inputSpec.get_parameter().get_parameter_name());
        const auto& inputName = inputSpec.get_parameter().get_arg().get_name();
        const auto& weightName = inputSpec.get_parameter().get_parameter_name();
        inputsToParameters_.emplace(inputName, weightName);
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::BUFFER: {
        const bool isPersistent = inputSpec.get_buffer().get_persistent();
        const std::string& bufferName =
            inputSpec.get_buffer().get_buffer_name();
        if (isPersistent) {
          buffers_.push_back(bufferName);
        } else {
          nonPersistentBuffers_.push_back(bufferName);
        }
        const auto& inputName = inputSpec.get_buffer().get_arg().get_name();
        const auto& weightName = inputSpec.get_buffer().get_buffer_name();
        inputsToBuffers_.emplace(inputName, weightName);
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::TENSOR_CONSTANT: {
        tensorConstants_.push_back(
            inputSpec.get_tensor_constant().get_tensor_constant_name());
        const auto& inputName =
            inputSpec.get_tensor_constant().get_arg().get_name();
        const auto& weightName =
            inputSpec.get_tensor_constant().get_tensor_constant_name();

        inputsToTensorConstants_.emplace(inputName, weightName);
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::CUSTOM_OBJ: {
        customObjs_.push_back(inputSpec.get_custom_obj().get_custom_obj_name());
        inputsToCustomObjs_.insert(
            {inputSpec.get_custom_obj().get_arg().get_name(),
             inputSpec.get_custom_obj().get_custom_obj_name()});
        break;
      }
      case torch::_export::InputSpec::Tag::CONSTANT_INPUT: {
        constantInputs_.push_back(inputSpec.get_constant_input().get_name());
        break;
      }
      case torch::_export::InputSpec::Tag::TOKEN: {
        throw std::runtime_error("Token inputs not implemented yet");
      }
      default:
        LOG(FATAL) << "Got empty thrift argument";
        break;
    }
  }

  std::string lossOutput;
  for (const torch::_export::OutputSpec& outputSpec :
       storage.get_output_specs()) {
    switch (outputSpec.tag()) {
      case torch::_export::OutputSpec::Tag::LOSS_OUTPUT:
        lossOutput_ = outputSpec.get_loss_output().get_arg().get_name();
        break;
      case torch::_export::OutputSpec::Tag::USER_OUTPUT:
        if (isSymbolicOutput(outputSpec.get_user_output().get_arg().tag())) {
          switch (outputSpec.get_user_output().get_arg().tag()) {
            case torch::_export::Argument::Tag::AS_TENSOR: {
              userOutputs_.emplace_back(outputSpec.get_user_output()
                                            .get_arg()
                                            .get_as_tensor()
                                            .get_name());
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_INT: {
              userOutputs_.emplace_back(outputSpec.get_user_output()
                                            .get_arg()
                                            .get_as_sym_int()
                                            .get_as_name());
              break;
            }
            default: {
              LOG(FATAL) << "Unsupported symbolic user output type ";
            }
          }
        } else {
          // for constant outputs, we don't have a name
          userOutputs_.emplace_back(std::nullopt);
        }
        break;
      case torch::_export::OutputSpec::Tag::BUFFER_MUTATION:
        buffersToMutate_.insert(
            {outputSpec.get_buffer_mutation().get_arg().get_name(),
             outputSpec.get_buffer_mutation().get_buffer_name()});
        break;
      case torch::_export::OutputSpec::Tag::GRADIENT_TO_PARAMETER:
        gradientsToParameters_.insert(
            {outputSpec.get_gradient_to_parameter().get_arg().get_name(),
             outputSpec.get_gradient_to_parameter().get_parameter_name()});
        break;
      case torch::_export::OutputSpec::Tag::GRADIENT_TO_USER_INPUT:
        gradientsToUserInputs_.insert(
            {outputSpec.get_gradient_to_user_input().get_arg().get_name(),
             outputSpec.get_gradient_to_user_input().get_user_input_name()});
        break;
      case torch::_export::OutputSpec::Tag::USER_INPUT_MUTATION:
        userInputsToMutate_.insert(
            {outputSpec.get_user_input_mutation().get_arg().get_name(),
             outputSpec.get_user_input_mutation().get_user_input_name()});
        break;
      case torch::_export::OutputSpec::Tag::TOKEN: {
        throw std::runtime_error("Token outputs not implemented yet");
      }
      default:
        LOG(FATAL) << "Got empty thrift argument";
    }
  }

  auto keys_of = [&](const std::unordered_map<std::string, std::string>& dict) {
    std::vector<std::string_view> keys;
    keys.reserve(dict.size());
    for (const auto& [key, _] : dict) {
      keys.emplace_back(key);
    }
    return keys;
  };

  auto get_valid = [&](const std::vector<std::optional<std::string>>& outputs) {
    std::vector<std::string> validOutputs;
    for (const auto& output : outputs) {
      if (output.has_value()) {
        validOutputs.push_back(*output);
      } else {
        validOutputs.emplace_back("Constant");
      }
    }
    return validOutputs;
  };

  VLOG(1) << fmt::format("[{}]", fmt::join(userInputs_, ", "));
  VLOG(1) << fmt::format("[{}]", fmt::join(keys_of(inputsToParameters_), ", "));
  VLOG(1) << fmt::format("[{}]", fmt::join(keys_of(inputsToBuffers_), ", "));
  VLOG(1) << fmt::format(
      "[{}]", fmt::join(keys_of(inputsToTensorConstants_), ", "));
  VLOG(1) << fmt::format("[{}]", fmt::join(get_valid(userOutputs_), ", "));
  VLOG(1) << fmt::format("[{}]", fmt::join(keys_of(buffersToMutate_), ", "));
  VLOG(1) << fmt::format(
      "[{}]", fmt::join(keys_of(gradientsToParameters_), ", "));
  VLOG(1) << fmt::format(
      "[{}]", fmt::join(keys_of(gradientsToUserInputs_), ", "));
}

std::set<std::string> GraphSignature::inputNames() const {
  std::set<std::string> ret;
  for (const auto& name : userInputs()) {
    ret.insert(name);
  }
  for (const auto& [inputName, _] : inputsToParameters()) {
    ret.insert(inputName);
  }
  for (const auto& [inputName, _] : inputsToBuffers()) {
    ret.insert(inputName);
  }
  for (const auto& [inputName, _] : inputsToTensorConstants()) {
    ret.insert(inputName);
  }
  for (const auto& [inputName, _] : inputsToCustomObjs()) {
    ret.insert(inputName);
  }
  return ret;
}

std::set<std::optional<std::string>> GraphSignature::outputNames() const {
  std::set<std::optional<std::string>> ret;
  for (const auto& name : userOutputs()) {
    ret.insert(name);
  }
  for (const auto& [outputName, _] : buffersToMutate()) {
    ret.insert(outputName);
  }
  for (const auto& [outputName, _] : userInputsToMutate()) {
    ret.insert(outputName);
  }
  if (hasBackward()) {
    if (!gradientsToParameters().empty()) {
      for (const auto& [outputName, _] : gradientsToParameters()) {
        ret.insert(outputName);
      }
    }
    if (!gradientsToUserInputs().empty()) {
      for (const auto& [outputName, _] : gradientsToUserInputs()) {
        ret.insert(outputName);
      }
    }
    if (!lossOutput().empty()) {
      ret.insert(lossOutput());
    }
  }
  return ret;
}

void GraphSignature::lint(
    const std::set<std::string>& graphInputs,
    const std::set<std::string>& graphOutputs) const {
  checkInputNames(inputNames(), graphInputs);
  checkOutputNames(outputNames(), graphOutputs);
}

void GraphSignature::replaceAllUses(
    std::string_view old,
    std::string_view replacement) {
  if (old == replacement) {
    return;
  }
  for (auto& name : userOutputs_) {
    if (name == old) {
      name = replacement;
    }
  }
  replaceInMap(buffersToMutate_, old, replacement);
  if (hasBackward()) {
    replaceInMap(gradientsToParameters_, old, replacement);
    replaceInMap(gradientsToUserInputs_, old, replacement);
    if (old == lossOutput_) {
      lossOutput_ = replacement;
    }
  }
}

std::ostream& operator<<(std::ostream& out, const GraphSignature& sig) {
  out << "inputsToParameters: {\n";
  for (const auto& [inputName, paramName] : sig.inputsToParameters()) {
    out << "\t" << inputName << " : " << paramName << std::endl;
  }
  out << "}\n";

  out << "inputsToBuffers: {\n";
  for (const auto& [inputName, bufferName] : sig.inputsToBuffers()) {
    out << "\t" << inputName << " : " << bufferName << std::endl;
  }
  out << "}\n";

  out << "inputsToTensorConstants: {\n";
  for (const auto& [inputName, tensorConstantName] :
       sig.inputsToTensorConstants()) {
    out << "\t" << inputName << " : " << tensorConstantName << std::endl;
  }
  out << "}\n";

  return out;
}

} // namespace torch::nativert
