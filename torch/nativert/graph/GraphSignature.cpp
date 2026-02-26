#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <array>
#include <iostream>

#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/graph/GraphSignature.h>

namespace torch::nativert {

namespace {

bool isSymbolicOutput(torch::_export::Argument::Tag t) {
  switch (t) {
    case torch::_export::Argument::Tag::AS_TENSOR:
    case torch::_export::Argument::Tag::AS_TENSORS:
    case torch::_export::Argument::Tag::AS_NESTED_TENSORS:
    case torch::_export::Argument::Tag::AS_OPTIONAL_TENSOR:
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
  switch (inputSpec.tag()) {
    case torch::_export::InputSpec::Tag::PARAMETER:
      return std::make_pair(
          inputSpec.get_parameter().get_arg().get_name(), "PARAMETER");
      break;
    case torch::_export::InputSpec::Tag::BUFFER:
      return std::make_pair(
          inputSpec.get_buffer().get_arg().get_name(), "BUFFER");
      break;
    case torch::_export::InputSpec::Tag::TENSOR_CONSTANT:
      return std::make_pair(
          inputSpec.get_tensor_constant().get_arg().get_name(),
          "TENSOR_CONSTANT");
      break;
    case torch::_export::InputSpec::Tag::CUSTOM_OBJ:
      return std::make_pair(
          inputSpec.get_custom_obj().get_arg().get_name(), "CUSTOM_OBJ");
      break;
    case torch::_export::InputSpec::Tag::USER_INPUT:
      if (inputSpec.get_user_input().get_arg().tag() ==
          torch::_export::Argument::Tag::AS_TENSOR) {
        return std::make_pair(
            inputSpec.get_user_input().get_arg().get_as_tensor().get_name(),
            "USER_INPUT");
      } else if (
          inputSpec.get_user_input().get_arg().tag() ==
          torch::_export::Argument::Tag::AS_CUSTOM_OBJ) {
        return std::make_pair(
            inputSpec.get_user_input().get_arg().get_as_custom_obj().get_name(),
            "USER_INPUT");
      } else {
        TORCH_CHECK(false, "Unsupported USER_INPUT argument type.");
      }
      break;
    case torch::_export::InputSpec::Tag::CONSTANT_INPUT:
      return std::make_pair(
          inputSpec.get_constant_input().get_name(), "CONSTANT_INPUT");
      break;
    case torch::_export::InputSpec::Tag::TOKEN:
      TORCH_CHECK(false, "Token inputs not implemented yet.");
    default:
      TORCH_CHECK(false, "Unknown InputSpec tag encountered.");
  }
}

void checkInputOrders(
    const std::vector<torch::_export::InputSpec>& inputSpecs) {
  // Map each tag to its index in the expected order
  static constexpr std::
      array<std::pair<torch::_export::InputSpec::Tag, uint32_t>, 5>
          tagOrderArray = {
              {{torch::_export::InputSpec::Tag::TOKEN, 0},
               {torch::_export::InputSpec::Tag::PARAMETER, 1},
               {torch::_export::InputSpec::Tag::BUFFER, 2},
               {torch::_export::InputSpec::Tag::TENSOR_CONSTANT, 3},
               {torch::_export::InputSpec::Tag::CUSTOM_OBJ, 4}}};
  uint32_t currentOrderIndex = 0;
  bool seenNonPersistentBuffer = false;
  for (const auto& inputSpec : inputSpecs) {
    if (inputSpec.tag() == torch::_export::InputSpec::Tag::USER_INPUT ||
        inputSpec.tag() == torch::_export::InputSpec::Tag::CONSTANT_INPUT) {
      continue;
    }
    auto it = std::find_if(
        tagOrderArray.begin(),
        tagOrderArray.end(),
        [&inputSpec](const auto& pair) {
          return pair.first == inputSpec.tag();
        });
    TORCH_CHECK(
        it != tagOrderArray.end(), "Unknown InputSpec tag encountered.");
    uint32_t tagIndex = it->second;
    if (tagIndex < currentOrderIndex) {
      auto [argName, tagName] = getSpecDetails(inputSpec);
      TORCH_CHECK(
          false,
          fmt::format(
              "Input arg {} with InputSpec {} is out of order!",
              argName,
              tagName));
    }
    currentOrderIndex = tagIndex;
    // Additional check for buffers
    if (inputSpec.tag() == torch::_export::InputSpec::Tag::BUFFER) {
      if (!inputSpec.get_buffer().get_persistent()) {
        seenNonPersistentBuffer = true;
      } else {
        TORCH_CHECK(
            !seenNonPersistentBuffer,
            "Persistent buffers must come before non-persistent buffers.");
      }
    }
  }
}

void checkInputNames(
    const c10::FastSet<std::string>& sigNames,
    const c10::FastSet<std::string>& graphNames) {
  if (sigNames == graphNames) {
    return;
  }

  std::string errorMsg = fmt::format(
      "Error: Value name difference detected between graph signature and graph nodes:\n"
      "Signature value names:\n[{}]\n"
      "Graph node names:\n[{}]",
      fmt::join(sigNames, ", "),
      fmt::join(graphNames, ", "));
  TORCH_CHECK(false, errorMsg);
}

void checkOutputNames(
    const c10::FastSet<std::optional<std::string>>& sigNames,
    const c10::FastSet<std::string>& graphNames) {
  std::vector<std::string> validNames;
  for (const auto& nameOpt : sigNames) {
    if (nameOpt.has_value()) {
      validNames.push_back(*nameOpt);
    }
  }

  for (const auto& name : validNames) {
    if (graphNames.find(name) == graphNames.end()) {
      std::string errorMsg = fmt::format(
          "Error: Value name difference detected between graph signature and graph nodes:\n"
          "Signature value names:\n[{}]\n"
          "Graph node names:\n[{}]",
          fmt::join(validNames, ", "),
          fmt::join(graphNames, ", "));
      TORCH_CHECK(false, errorMsg);
    }
  }
}

void replaceInMap(
    c10::FastMap<std::string, std::string>& map,
    std::string_view old,
    std::string_view replacement) {
  auto it = map.find(std::string{old});
  if (it == map.end()) {
    return;
  }
  std::string value = std::move(it->second);
  map.erase(it);
  map.emplace(replacement, std::move(value));
}

} // namespace

GraphSignature::GraphSignature(const torch::_export::GraphSignature& storage) {
  checkInputOrders(storage.get_input_specs());

  for (const torch::_export::InputSpec& inputSpec : storage.get_input_specs()) {
    switch (inputSpec.tag()) {
      case torch::_export::InputSpec::Tag::USER_INPUT: {
        const auto& userInputArg = inputSpec.get_user_input().get_arg();
        if (userInputArg.tag() == torch::_export::Argument::Tag::AS_TENSOR) {
          userInputs_.emplace_back(userInputArg.get_as_tensor().get_name());
        } else if (
            userInputArg.tag() ==
            torch::_export::Argument::Tag::AS_CUSTOM_OBJ) {
          userInputs_.emplace_back(userInputArg.get_as_custom_obj().get_name());
        } else if (
            userInputArg.tag() == torch::_export::Argument::Tag::AS_TENSORS) {
          // Handle list of tensors
          for (const auto& tensor : userInputArg.get_as_tensors()) {
            userInputs_.emplace_back(tensor.get_name());
          }
        } else if (
            userInputArg.tag() ==
            torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS) {
          // Handle list of optional tensors
          for (const auto& optTensor : userInputArg.get_as_optional_tensors()) {
            if (optTensor.tag() ==
                torch::_export::OptionalTensorArgument::Tag::AS_TENSOR) {
              userInputs_.emplace_back(optTensor.get_as_tensor().get_name());
            }
            // Skip None tensors
          }
        } else if (
            userInputArg.tag() ==
            torch::_export::Argument::Tag::AS_OPTIONAL_TENSOR) {
          // Handle single optional tensor
          const auto& optTensor = userInputArg.get_as_optional_tensor();
          if (optTensor.tag() ==
              torch::_export::OptionalTensorArgument::Tag::AS_TENSOR) {
            userInputs_.emplace_back(optTensor.get_as_tensor().get_name());
          }
          // Skip if None
        } else if (
            userInputArg.tag() == torch::_export::Argument::Tag::AS_SYM_INT) {
          // Handle symbolic int input
          const auto& symInt = userInputArg.get_as_sym_int();
          if (symInt.tag() == torch::_export::SymIntArgument::Tag::AS_NAME) {
            userInputs_.emplace_back(symInt.get_as_name());
          }
          // Skip AS_INT (constant) symints
        } else if (
            userInputArg.tag() == torch::_export::Argument::Tag::AS_SYM_INTS) {
          // Handle list of symbolic ints
          for (const auto& symInt : userInputArg.get_as_sym_ints()) {
            if (symInt.tag() == torch::_export::SymIntArgument::Tag::AS_NAME) {
              userInputs_.emplace_back(symInt.get_as_name());
            }
            // Skip AS_INT (constant) symints
          }
        } else if (
            userInputArg.tag() == torch::_export::Argument::Tag::AS_NONE ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_INT ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_INTS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_FLOAT ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_FLOATS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_BOOL ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_BOOLS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_STRING ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_STRINGS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_SYM_BOOL ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_SYM_BOOLS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_SYM_FLOAT ||
            userInputArg.tag() ==
                torch::_export::Argument::Tag::AS_SYM_FLOATS ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_INT_LISTS ||
            userInputArg.tag() ==
                torch::_export::Argument::Tag::AS_SCALAR_TYPE ||
            userInputArg.tag() ==
                torch::_export::Argument::Tag::AS_MEMORY_FORMAT ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_LAYOUT ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_DEVICE ||
            userInputArg.tag() == torch::_export::Argument::Tag::AS_COMPLEX) {
          // Non-tensor inputs are constant values, skip them for now
          // These don't map to named graph inputs in the same way tensors do
        } else {
          // TODO: handle other types
          TORCH_CHECK(false, "Non tensor inputs not implemented yet.");
        }
        break;
      }
      case torch::_export::InputSpec::Tag::PARAMETER: {
        numParameters_++;
        const auto& inputName = inputSpec.get_parameter().get_arg().get_name();
        const auto& weightName = inputSpec.get_parameter().get_parameter_name();
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::BUFFER: {
        const bool isPersistent = inputSpec.get_buffer().get_persistent();
        const auto& inputName = inputSpec.get_buffer().get_arg().get_name();
        const auto& weightName = inputSpec.get_buffer().get_buffer_name();
        if (isPersistent) {
          numPersistentBuffers_++;
        } else {
          numNonPersistentBuffers_++;
        }
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::TENSOR_CONSTANT: {
        numTensorConstants_++;
        const auto& inputName =
            inputSpec.get_tensor_constant().get_arg().get_name();
        const auto& weightName =
            inputSpec.get_tensor_constant().get_tensor_constant_name();
        inputsToWeights_.emplace_back(inputName, weightName);
        break;
      }
      case torch::_export::InputSpec::Tag::CUSTOM_OBJ: {
        numCustomObjs_++;
        const auto& inputName = inputSpec.get_custom_obj().get_arg().get_name();
        const auto& customObjName =
            inputSpec.get_custom_obj().get_custom_obj_name();
        inputsToCustomObjs_.emplace_back(inputName, customObjName);
        break;
      }
      case torch::_export::InputSpec::Tag::CONSTANT_INPUT: {
        break;
      }
      case torch::_export::InputSpec::Tag::TOKEN: {
        TORCH_CHECK(false, "Token inputs not implemented yet.");
      }
      default:
        TORCH_CHECK(false, "Unknown InputSpec tag encountered.");
        break;
    }
  }

  for (const torch::_export::OutputSpec& outputSpec :
       storage.get_output_specs()) {
    switch (outputSpec.tag()) {
      case torch::_export::OutputSpec::Tag::LOSS_OUTPUT:
        lossOutput_ = outputSpec.get_loss_output().get_arg().get_name();
        break;
      case torch::_export::OutputSpec::Tag::USER_OUTPUT: {
        const auto& userOutputArg = outputSpec.get_user_output().get_arg();
        if (isSymbolicOutput(userOutputArg.tag())) {
          switch (userOutputArg.tag()) {
            case torch::_export::Argument::Tag::AS_TENSOR: {
              userOutputs_.emplace_back(
                  userOutputArg.get_as_tensor().get_name());
              break;
            }
            case torch::_export::Argument::Tag::AS_TENSORS: {
              // Handle list of tensors - each tensor is a separate output
              for (const auto& tensor : userOutputArg.get_as_tensors()) {
                userOutputs_.emplace_back(tensor.get_name());
              }
              break;
            }
            case torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS: {
              // Handle list of optional tensors
              for (const auto& optTensor :
                   userOutputArg.get_as_optional_tensors()) {
                if (optTensor.tag() ==
                    torch::_export::OptionalTensorArgument::Tag::AS_TENSOR) {
                  userOutputs_.emplace_back(
                      optTensor.get_as_tensor().get_name());
                } else {
                  // None tensor - no name
                  userOutputs_.emplace_back(std::nullopt);
                }
              }
              break;
            }
            case torch::_export::Argument::Tag::AS_OPTIONAL_TENSOR: {
              // Handle single optional tensor
              const auto& optTensor = userOutputArg.get_as_optional_tensor();
              if (optTensor.tag() ==
                  torch::_export::OptionalTensorArgument::Tag::AS_TENSOR) {
                userOutputs_.emplace_back(optTensor.get_as_tensor().get_name());
              } else {
                // None tensor - no name
                userOutputs_.emplace_back(std::nullopt);
              }
              break;
            }
            case torch::_export::Argument::Tag::AS_CUSTOM_OBJ: {
              userOutputs_.emplace_back(
                  userOutputArg.get_as_custom_obj().get_name());
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_INT: {
              userOutputs_.emplace_back(
                  userOutputArg.get_as_sym_int().get_as_name());
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_INTS: {
              for (const auto& symInt : userOutputArg.get_as_sym_ints()) {
                if (symInt.tag() ==
                    torch::_export::SymIntArgument::Tag::AS_NAME) {
                  userOutputs_.emplace_back(symInt.get_as_name());
                }
                // Skip AS_INT (constant) symints
              }
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_BOOL: {
              userOutputs_.emplace_back(
                  userOutputArg.get_as_sym_bool().get_as_name());
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_BOOLS: {
              for (const auto& symBool : userOutputArg.get_as_sym_bools()) {
                if (symBool.tag() ==
                    torch::_export::SymBoolArgument::Tag::AS_NAME) {
                  userOutputs_.emplace_back(symBool.get_as_name());
                }
                // Skip AS_BOOL (constant) symbools
              }
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_FLOAT: {
              // SymFloat doesn't have get_as_name in all versions
              // For now, treat as unnamed symbolic output
              userOutputs_.emplace_back(std::nullopt);
              break;
            }
            case torch::_export::Argument::Tag::AS_SYM_FLOATS: {
              // SymFloats - treat each as unnamed for now
              for (size_t i = 0; i < userOutputArg.get_as_sym_floats().size();
                   ++i) {
                userOutputs_.emplace_back(std::nullopt);
              }
              break;
            }
            default: {
              TORCH_CHECK(
                  false, "Unsupported symbolic user output type encountered.");
            }
          }
        } else {
          // for constant outputs, we don't have a name
          userOutputs_.emplace_back(std::nullopt);
        }
        break;
      }
      case torch::_export::OutputSpec::Tag::BUFFER_MUTATION:
        buffersToMutate_.emplace(
            outputSpec.get_buffer_mutation().get_arg().get_name(),
            outputSpec.get_buffer_mutation().get_buffer_name());
        break;
      case torch::_export::OutputSpec::Tag::GRADIENT_TO_PARAMETER:
        gradientsToParameters_.emplace(
            outputSpec.get_gradient_to_parameter().get_arg().get_name(),
            outputSpec.get_gradient_to_parameter().get_parameter_name());
        break;
      case torch::_export::OutputSpec::Tag::GRADIENT_TO_USER_INPUT:
        gradientsToUserInputs_.emplace(
            outputSpec.get_gradient_to_user_input().get_arg().get_name(),
            outputSpec.get_gradient_to_user_input().get_user_input_name());
        break;
      case torch::_export::OutputSpec::Tag::USER_INPUT_MUTATION:
        userInputsToMutate_.emplace(
            outputSpec.get_user_input_mutation().get_arg().get_name(),
            outputSpec.get_user_input_mutation().get_user_input_name());
        break;
      case torch::_export::OutputSpec::Tag::TOKEN: {
        TORCH_CHECK(false, "Token outputs not implemented yet.");
      }
      default:
        TORCH_CHECK(false, "Unknown OutputSpec tag encountered.");
    }
  }

  if (FLAGS_caffe2_log_level > 2) {
    std::cout << *this << '\n';
  }
}

c10::FastSet<std::string> GraphSignature::inputNames() const {
  c10::FastSet<std::string> ret;
  size_t numInputs = userInputs().size() + inputsToWeights().size() +
      inputsToCustomObjs().size();
  ret.reserve(numInputs);
  for (const auto& name : userInputs()) {
    ret.insert(name);
  }
  for (const auto& [inputName, _] : inputsToWeights()) {
    ret.insert(inputName);
  }
  for (const auto& [inputName, _] : inputsToCustomObjs()) {
    ret.insert(inputName);
  }
  return ret;
}

c10::FastSet<std::optional<std::string>> GraphSignature::outputNames() const {
  c10::FastSet<std::optional<std::string>> ret;
  size_t numOutputs = userOutputs().size() + buffersToMutate().size() +
      userInputsToMutate().size() +
      (hasBackward() ? gradientsToParameters().size() +
               gradientsToUserInputs().size() + (lossOutput().empty() ? 0 : 1)
                     : 0);
  ret.reserve(numOutputs);
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
    const c10::FastSet<std::string>& graphInputs,
    const c10::FastSet<std::string>& graphOutputs) const {
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
  if (!sig.inputsToParameters().empty()) {
    out << "inputsToParameters: {\n";
    for (const auto& [inputName, paramName] : sig.inputsToParameters()) {
      out << '\t' << inputName << " : " << paramName << '\n';
    }
    out << "}\n";
  }
  if (!sig.inputsToBuffers().empty()) {
    out << "inputsToBuffers: {\n";
    for (const auto& [inputName, bufferName] : sig.inputsToBuffers()) {
      out << '\t' << inputName << " : " << bufferName << '\n';
    }
    out << "}\n";
  }
  if (!sig.inputsToTensorConstants().empty()) {
    out << "inputsToTensorConstants: {\n";
    for (const auto& [inputName, tensorConstantName] :
         sig.inputsToTensorConstants()) {
      out << '\t' << inputName << " : " << tensorConstantName << '\n';
    }
    out << "}\n";
  }
  if (!sig.inputsToCustomObjs().empty()) {
    out << "inputsToCustomObjs: {\n";
    for (const auto& [inputName, customObjName] : sig.inputsToCustomObjs()) {
      out << '\t' << inputName << " : " << customObjName << '\n';
    }
    out << "}\n";
  }
  if (!sig.userOutputs().empty()) {
    out << "userOutputs: {\n";
    for (const auto& outputName : sig.userOutputs()) {
      out << '\t' << outputName.value_or("Constant") << '\n';
    }
    out << "}\n";
  }
  if (!sig.buffersToMutate().empty()) {
    out << "buffersToMutate: {\n";
    for (const auto& [outputName, mutatedBufferName] : sig.buffersToMutate()) {
      out << '\t' << outputName << " : " << mutatedBufferName << '\n';
    }
    out << "}\n";
  }
  if (!sig.userInputsToMutate().empty()) {
    out << "userInputsToMutate: {\n";
    for (const auto& [outputName, mutatedUserInputName] :
         sig.userInputsToMutate()) {
      out << '\t' << outputName << " : " << mutatedUserInputName << '\n';
    }
    out << "}\n";
  }
  if (sig.hasBackward()) {
    if (!sig.gradientsToParameters().empty()) {
      out << "gradientsToParameters: {\n";
      for (const auto& [outputName, paramName] : sig.gradientsToParameters()) {
        out << '\t' << outputName << " : " << paramName << '\n';
      }
      out << "}\n";
    }
    if (!sig.gradientsToUserInputs().empty()) {
      out << "gradientsToUserInputs: {\n";
      for (const auto& [outputName, userInputName] :
           sig.gradientsToUserInputs()) {
        out << '\t' << outputName << " : " << userInputName << '\n';
      }
      out << "}\n";
    }
    out << "lossOutput: " << sig.lossOutput() << '\n';
  }
  return out;
}

} // namespace torch::nativert
