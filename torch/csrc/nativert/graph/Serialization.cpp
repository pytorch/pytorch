#include "torch/csrc/nativert/graph/Serialization.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

namespace torch::nativert {

namespace {

std::unique_ptr<Graph> jsonToSubgraph(
    const torch::_export::Graph& thriftGraph,
    const torch::_export::GraphSignature* signature,
    bool loadNodeMetadata);

Value* symbolicToValue(
    const torch::_export::Argument& arg,
    Graph& graph,
    Node* insertBefore) {
  switch (arg.tag()) {
    case torch::_export::Argument::Tag::AS_TENSOR:
      return graph.getValue(arg.get_as_tensor().get_name());
    case torch::_export::Argument::Tag::AS_TENSORS: {
      // Need to insert a list pack node
      std::vector<Value*> listValue;
      for (const auto& listEl : arg.get_as_tensors()) {
        listValue.push_back(graph.getValue(listEl.get_name()));
      }
      auto listPack = graph.createListPack(std::move(listValue), Type::Tensor);
      return graph.insertBefore(listPack, insertBefore)->outputs()[0];
    }
    case torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS: {
      // Need to insert a list pack node
      std::vector<Value*> listValue;
      for (const auto& listEl : arg.get_as_optional_tensors()) {
        switch (listEl.tag()) {
          case torch::_export::OptionalTensorArgument::Tag::AS_TENSOR: {
            listValue.push_back(
                graph.getValue(listEl.get_as_tensor().get_name()));
            break;
          }
          case torch::_export::OptionalTensorArgument::Tag::AS_NONE: {
            listValue.push_back(
                graph.addValue(std::nullopt, Type::None, nullptr));
            break;
          }
          default:
            throw std::runtime_error(fmt::format(
                "Unknown OptionalTensorArgument type: {}",
                torch::_export::printEnum(listEl.tag())));
        }
      }
      auto listPack = graph.createOptionalListPack(std::move(listValue));
      return graph.insertBefore(listPack, insertBefore)->outputs()[0];
    }
    case torch::_export::Argument::Tag::AS_SYM_INT: {
      return graph.getValue(arg.get_as_sym_int().get_as_name());
    }
    case torch::_export::Argument::Tag::AS_SYM_INTS: {
      // Need to insert a list pack node
      std::vector<Value*> listValue;
      for (const auto& listEl : arg.get_as_sym_ints()) {
        switch (listEl.tag()) {
          case torch::_export::SymIntArgument::Tag::AS_NAME: {
            listValue.push_back(graph.getValue(listEl.get_as_name()));
            break;
          }
          case torch::_export::SymIntArgument::Tag::AS_INT: {
            // These are concrete int values in the SymIntList, e.g [s0, 8]
            // We convert them into a constant Value in graph. These value
            // doesn't have producer node
            int value = listEl.get_as_int();
            Value* symintValue = graph.createConstantSymIntValue(value);
            listValue.push_back(symintValue);
            break;
          }
          default:
            throw std::runtime_error(fmt::format(
                "Unknown SymIntArgument type: {}",
                torch::_export::printEnum(listEl.tag())));
        }
      }
      auto listPack = graph.createListPack(std::move(listValue), Type::SymInt);
      return graph.insertBefore(listPack, insertBefore)->outputs()[0];
    }
    case torch::_export::Argument::Tag::AS_CUSTOM_OBJ: {
      return graph.getValue(arg.get_as_custom_obj().get_name());
    }
    case torch::_export::Argument::Tag::AS_SYM_BOOL: {
      return graph.getValue(arg.get_as_sym_bool().get_as_name());
    }
    case torch::_export::Argument::Tag::AS_SYM_FLOAT: {
      return graph.getValue(arg.get_as_sym_float().get_as_name());
    }
    default:
      throw std::runtime_error(fmt::format(
          "This function should only be called for symbolics, got \'{}\' instead",
          torch::_export::printEnum(arg.tag())));
  }
}

std::pair<
    std::vector<torch::_export::InputSpec>,
    std::vector<torch::_export::Argument>>
enforceInputOrder(
    const std::vector<torch::_export::InputSpec>& inputSpecs,
    const std::vector<torch::_export::Argument>& graphInputs) {
  // Enforce the order of inputSpecs and graphInputs to be the following:
  // 1. token
  // 2. parameter
  // 3. persistent buffer, non-persistent buffer
  // 4. tensor_constant
  // 5. custom_obj
  // 6. user_input/constant_input
  std::vector<torch::_export::InputSpec> reorderedInputSpecs;
  std::vector<torch::_export::Argument> reorderedGraphInputs;
  std::vector<torch::_export::InputSpec::Tag> desiredOrder = {
      torch::_export::InputSpec::Tag::TOKEN,
      torch::_export::InputSpec::Tag::PARAMETER,
      torch::_export::InputSpec::Tag::BUFFER,
      torch::_export::InputSpec::Tag::TENSOR_CONSTANT,
      torch::_export::InputSpec::Tag::CUSTOM_OBJ};

  auto reorder = [&](auto condition) {
    for (size_t i = 0; i < inputSpecs.size(); ++i) {
      if (condition(inputSpecs[i])) {
        reorderedInputSpecs.push_back(inputSpecs[i]);
        reorderedGraphInputs.push_back(graphInputs[i]);
      }
    }
  };

  for (const auto& tag : desiredOrder) {
    if (tag == torch::_export::InputSpec::Tag::BUFFER) {
      // Add persistent buffers first, then non-persistent
      reorder([&](const auto& spec) {
        return spec.tag() == tag && spec.get_buffer().get_persistent();
      });
      reorder([&](const auto& spec) {
        return spec.tag() == tag && !spec.get_buffer().get_persistent();
      });
    } else {
      reorder([&](const auto& spec) { return spec.tag() == tag; });
    }
  }

  // Append USER_INPUT and CONSTANT_INPUT without reordering
  for (size_t i = 0; i < inputSpecs.size(); ++i) {
    auto tag = inputSpecs[i].tag();
    if (tag == torch::_export::InputSpec::Tag::USER_INPUT ||
        tag == torch::_export::InputSpec::Tag::CONSTANT_INPUT) {
      reorderedInputSpecs.push_back(inputSpecs[i]);
      reorderedGraphInputs.push_back(graphInputs[i]);
    }
  }
  return {std::move(reorderedInputSpecs), std::move(reorderedGraphInputs)};
}

std::unique_ptr<Graph> jsonToSubgraph(
    const torch::_export::Graph& jsonGraph,
    const torch::_export::GraphSignature* signature,
    bool loadNodeMetadata) {
  auto graphInputs = jsonGraph.get_inputs();
  auto graph = Graph::createGraph();

  if (signature) {
    // enforcing the order signature inputspecs and graph inputs
    auto inputSpecs = signature->get_input_specs();

    auto [reorderedInputSpecs, reorderedGraphInputs] =
        enforceInputOrder(inputSpecs, graphInputs);

    graphInputs = std::move(reorderedGraphInputs);
    auto reorderedSignature = *signature;
    reorderedSignature.set_input_specs(reorderedInputSpecs);
    graph->setSignature(GraphSignature{reorderedSignature});
  }

  for (const auto& input : graphInputs) {
    if (isSymbolic(input)) {
      switch (input.tag()) {
        case torch::_export::Argument::Tag::AS_TENSOR: {
          const auto& asTensor = input.get_as_tensor();
          const auto& name = asTensor.get_name();
          graph->addInput(name, Type::Tensor);
          break;
        }
        case torch::_export::Argument::Tag::AS_CUSTOM_OBJ: {
          const auto& asCustomObj = input.get_as_custom_obj();
          const std::string& name = asCustomObj.get_name();
          const std::string& classFqn = asCustomObj.get_class_fqn();
          graph->addInput(name, Type(Type::CustomObj, classFqn));
          break;
        }
        default:
          throw std::runtime_error(fmt::format(
              "Unsupported graph input type {}",
              static_cast<int>(input.tag())));
      }
    } else {
      switch (input.tag()) {
        case torch::_export::Argument::Tag::AS_INT:
        case torch::_export::Argument::Tag::AS_FLOAT:
        case torch::_export::Argument::Tag::AS_STRING:
        case torch::_export::Argument::Tag::AS_BOOL:
        case torch::_export::Argument::Tag::AS_NONE: {
          // Constant graph inputs are specialized in the graph, here we simply
          // add a nullptr of Value to the graph input node.
          graph->addInput();
          break;
        }
        default:
          throw std::runtime_error(fmt::format(
              "Unsupported graph input type {}",
              static_cast<int>(input.tag())));
      }
    }
  }

  for (const auto& jsonNode : jsonGraph.get_nodes()) {
    auto node = graph->insertNode(
        jsonNode.get_target(),
        {},
        loadNodeMetadata ? jsonNode.get_metadata()
                         : std::unordered_map<std::string, std::string>());

    std::vector<NamedArgument> args;
    std::vector<Attribute> attributes;
    for (const auto& input : jsonNode.get_inputs()) {
      // We handle constants and symbolic inputs differently.
      const auto& arg = input.get_arg();
      if (isSymbolic(arg)) {
        // Symbolic values are made part of the inputs to the node
        node->addInput(NamedArgument{
            input.get_name(), symbolicToValue(input.get_arg(), *graph, node)});
      } else if (arg.tag() == torch::_export::Argument::Tag::AS_NONE) {
        node->addInput(NamedArgument{
            input.get_name(), graph->addValue(std::nullopt, Type::None, node)});
      } else {
        node->addAttribute(Attribute{
            input.get_name(),
            constantToValue(input.get_arg(), loadNodeMetadata)});
        // Constant values are added as "attributes" to the node.
      }
    }

    std::vector<Value*> outputs;
    std::vector<Value*> listUnpacksToCreate;
    for (const auto& output : jsonNode.get_outputs()) {
      switch (output.tag()) {
        case torch::_export::Argument::Tag::AS_NONE: {
          node->addOutput(Type::None);
          break;
        }
        case torch::_export::Argument::Tag::AS_TENSOR: {
          const auto name = output.get_as_tensor().get_name();
          node->addOutput(name, Type::Tensor);
          break;
        }
        case torch::_export::Argument::Tag::AS_TENSORS: {
          auto outputValue =
              node->addOutput(graph->getUniqueValueName(), Type::TensorList);

          Node* listUnpack =
              graph->insertNode("prim.ListUnpack", {{"input", outputValue}});
          for (const auto& arg : output.get_as_tensors()) {
            listUnpack->addOutput(arg.get_name(), Type::Tensor);
          }
          break;
        }
        case torch::_export::Argument::Tag::AS_SYM_INT: {
          const auto name = output.get_as_sym_int().get_as_name();
          node->addOutput(name, Type::SymInt);
          break;
        }
        case torch::_export::Argument::Tag::AS_SYM_INTS: {
          throw std::runtime_error(
              "SymInts NYI. We currently doesn't have op that produces SymInts as output");
        }
        case torch::_export::Argument::Tag::AS_SYM_BOOL: {
          const auto name = output.get_as_sym_bool().get_as_name();
          node->addOutput(name, Type::SymBool);
          break;
        }
        case torch::_export::Argument::Tag::AS_SYM_BOOLS: {
          throw std::runtime_error(
              "SymBools NYI. We currently doesn't have op that produces SymBools as output");
        }
        case torch::_export::Argument::Tag::AS_SYM_FLOAT: {
          const auto name = output.get_as_sym_float().get_as_name();
          node->addOutput(name, Type::SymFloat);
          break;
        }
        case torch::_export::Argument::Tag::AS_SYM_FLOATS: {
          throw std::runtime_error(
              "SymFloats NYI. We currently doesn't have op that produces SymFloats as output");
        }
        default:
          throw std::runtime_error(fmt::format(
              "disallowed output type {}",
              torch::_export::printEnum(output.tag())));
      }
    }
  }

  for (const auto& output : jsonGraph.get_outputs()) {
    // handle symbolic outputs and constant outputs differently
    if (isSymbolic(output)) {
      switch (output.tag()) {
        case torch::_export::Argument::Tag::AS_TENSOR: {
          const auto& asTensor = output.get_as_tensor();
          const auto& name = asTensor.get_name();
          Value* outputValue = graph->getValue(name);
          graph->addOutput(outputValue);
          break;
        }
        case torch::_export::Argument::Tag::AS_SYM_INT: {
          const auto& asSymInt = output.get_as_sym_int();
          TORCH_CHECK(
              asSymInt.tag() == torch::_export::SymIntArgument::Tag::AS_NAME);
          const auto& name = asSymInt.get_as_name();
          Value* outputValue = graph->getValue(name);
          graph->addOutput(outputValue);
          break;
        }
        default:
          throw std::runtime_error(fmt::format(
              "Unsupported graph output type: {}",
              static_cast<size_t>(output.tag())));
      }
    } else {
      Constant constValue = constantToValue(output, loadNodeMetadata);
      graph->addConstantOutput(constValue);
    }
  }

  auto jsonTensorValue = jsonGraph.get_tensor_values();

  if (!signature) {
    // For subgraphs we just need to derive a graph signature that only
    // contains user inputs and outputs, because we don't need to handle any
    // special semantics for them, e.g. mutation or gradients.
    torch::_export::GraphSignature sig;
    std::vector<torch::_export::InputSpec> inputSpecs;
    for (const auto& input : graph->inputs()) {
      torch::_export::Argument arg;
      if (input->type().kind() == Type::Tensor) {
        torch::_export::TensorArgument targ;
        targ.set_name(std::string{input->name()});
        arg.set_as_tensor(std::move(targ));
      } else {
        throw std::runtime_error(fmt::format(
            "Unsupported subgraph input type {}",
            fmt::streamed(input->type())));
      }
      torch::_export::UserInputSpec userInputSpec;
      userInputSpec.set_arg(std::move(arg));
      torch::_export::InputSpec inputSpec;
      inputSpec.set_user_input(std::move(userInputSpec));
      inputSpecs.push_back(std::move(inputSpec));
    }
    sig.set_input_specs(std::move(inputSpecs));

    std::vector<torch::_export::OutputSpec> outputSpecs;
    for (const auto& output : graph->outputs()) {
      torch::_export::Argument arg;
      if (output->type().kind() == Type::Tensor) {
        torch::_export::TensorArgument targ;
        targ.set_name(std::string{output->name()});
        arg.set_as_tensor(std::move(targ));
      } else {
        throw std::runtime_error(fmt::format(
            "Unsupported subgraph input type {}",
            fmt::streamed(output->type())));
      }
      torch::_export::UserOutputSpec userOutputSpec;
      userOutputSpec.set_arg(std::move(arg));
      torch::_export::OutputSpec outputSpec;
      outputSpec.set_user_output(std::move(userOutputSpec));
      outputSpecs.push_back(std::move(outputSpec));
    }
    sig.set_output_specs(std::move(outputSpecs));

    graph->setSignature(GraphSignature{sig});
  }

  // weightsTensorMeta are indexed by weight's name, not graph input's name
  std::unordered_map<std::string, torch::_export::TensorMeta> weightsTensorMeta;
  for (const auto& [inputName, weightName] :
       graph->signature().inputsToWeights()) {
    auto value = graph->getValue(inputName);
    if (value->type().kind() == Type::CustomObj) {
      // skip setting meta for non-tensor inputs
      continue;
    }

    auto it = jsonTensorValue.find(inputName);
    CHECK(it != jsonTensorValue.end())
        << "Missing tensor metadata for " << inputName
        << "in thriftGraph.tensorValue";
    weightsTensorMeta[weightName] = it->second;
  }
  graph->setWeightsMeta(weightsTensorMeta);

  graph->setTensorValuesMeta(jsonTensorValue);

  graph->finalize();

  graph->lint();
  return graph;
}

} // namespace

bool isSymbolic(const torch::_export::Argument& arg) {
  switch (arg.tag()) {
    case torch::_export::Argument::Tag::AS_TENSOR:
    case torch::_export::Argument::Tag::AS_TENSORS:
    case torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS:
    case torch::_export::Argument::Tag::AS_SYM_INT:
    case torch::_export::Argument::Tag::AS_SYM_INTS:
    case torch::_export::Argument::Tag::AS_SYM_BOOL:
    case torch::_export::Argument::Tag::AS_SYM_BOOLS:
    case torch::_export::Argument::Tag::AS_SYM_FLOAT:
    case torch::_export::Argument::Tag::AS_SYM_FLOATS:
    case torch::_export::Argument::Tag::AS_CUSTOM_OBJ:
      return true;
    default:
      return false;
  }
}

Constant constantToValue(
    const torch::_export::Argument& jsonArg,
    bool loadNodeMetadata) {
  switch (jsonArg.tag()) {
    case torch::_export::Argument::Tag::AS_TENSOR:
      throw std::runtime_error("Tensor is symbolic, shouldn't reach here");
    case torch::_export::Argument::Tag::AS_TENSORS:
      throw std::runtime_error("Tensor[] is symbolic, shouldn't reach here");
    case torch::_export::Argument::Tag::AS_OPTIONAL_TENSORS:
      throw std::runtime_error("Tensor?[] is symbolic, shouldn't reach here");
    case torch::_export::Argument::Tag::AS_NONE:
      return None();
    case torch::_export::Argument::Tag::AS_INT:
      return jsonArg.get_as_int();
    case torch::_export::Argument::Tag::AS_INTS: {
      std::vector<int64_t> ret;
      for (const auto& arg : jsonArg.get_as_ints()) {
        ret.push_back(arg);
      }
      return ret;
    }
    case torch::_export::Argument::Tag::AS_FLOAT:
      return jsonArg.get_as_float().get();
    case torch::_export::Argument::Tag::AS_FLOATS: {
      std::vector<double> ret;
      for (const auto& arg : jsonArg.get_as_floats()) {
        ret.push_back(arg.get());
      }
      return ret;
    }
    case torch::_export::Argument::Tag::AS_STRING:
      return jsonArg.get_as_string();
    case torch::_export::Argument::Tag::AS_STRINGS: {
      std::vector<std::string> ret;
      for (const auto& arg : jsonArg.get_as_strings()) {
        ret.push_back(arg);
      }
      return ret;
    }
    case torch::_export::Argument::Tag::AS_SYM_INT:
    case torch::_export::Argument::Tag::AS_SYM_INTS:
    case torch::_export::Argument::Tag::AS_SYM_BOOL:
    case torch::_export::Argument::Tag::AS_SYM_BOOLS:
      throw std::runtime_error(
          "Symint/Symbool Values are symbolic, shouldn't reach here");

    case torch::_export::Argument::Tag::AS_SCALAR_TYPE:
      return convertJsonScalarType(jsonArg.get_as_scalar_type());
    case torch::_export::Argument::Tag::AS_MEMORY_FORMAT:
      return convertJsonMemoryFormat(jsonArg.get_as_memory_format());
    case torch::_export::Argument::Tag::AS_LAYOUT:
      return convertJsonLayout(jsonArg.get_as_layout());
    case torch::_export::Argument::Tag::AS_DEVICE:
      return convertJsonDevice(jsonArg.get_as_device());
    case torch::_export::Argument::Tag::AS_BOOL:
      return jsonArg.get_as_bool();
    case torch::_export::Argument::Tag::AS_BOOLS: {
      std::vector<bool> ret;
      for (const auto& arg : jsonArg.get_as_bools()) {
        ret.push_back(arg);
      }
      return ret;
    }
    case torch::_export::Argument::Tag::AS_GRAPH: {
      return jsonToSubgraph(
          *jsonArg.get_as_graph().get_graph(), nullptr, loadNodeMetadata);
    }
    case torch::_export::Argument::Tag::AS_CUSTOM_OBJ:
      throw std::runtime_error("custom obj is dynamic, shouldn't reach here");
    case torch::_export::Argument::Tag::AS_OPERATOR:
      return jsonArg.get_as_operator();
    case torch::_export::Argument::Tag::AS_SYM_FLOAT: {
      throw std::runtime_error("sym float is not yet implemented");
    }
    case torch::_export::Argument::Tag::AS_SYM_FLOATS: {
      throw std::runtime_error("sym floats is not yet implemented");
    }
    default:
      throw std::runtime_error("Got unknown thrift argument");
  }
}

std::unique_ptr<Graph> jsonToGraph(
    const torch::_export::GraphModule& jsonGraphModule,
    bool loadNodeMetadata) {
  auto graph = jsonToSubgraph(
      jsonGraphModule.get_graph(),
      &jsonGraphModule.get_signature(),
      loadNodeMetadata);
  return graph;
}

} // namespace torch::nativert
