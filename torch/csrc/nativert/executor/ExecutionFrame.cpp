#include <c10/util/Logging.h>

#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/executor/ExecutionPlanner.h"

namespace torch::nativert {

ExecutionFrame::ExecutionFrame(const Graph& graph)
    : graph_(graph),
      allValues_(graph.numValues()),
      persistent_(graph.numValues()) {
  // load constant SymInts into execution frame
  for (const auto& [valueId, constSymintValue] :
       graph_.getConstantSymIntValues()) {
    setPersistentIValue(valueId, constSymintValue);
  }

  for (const Node& node : graph_.nodes()) {
    if (node.target() == "torch.ops.higher_order.run_const_graph") {
      const auto& const_graph =
          std::get<std::unique_ptr<Graph>>(node.attributes().at(0).value);
      for (size_t i = 0; i < node.outputs().size(); ++i) {
        foldedConstIds_[std::string{const_graph->outputs().at(i)->name()}] =
            node.outputs()[i]->id();
      }
    }
  }
}

ExecutionFrame::ExecutionFrame(const Graph& graph, const Weights& weights)
    : ExecutionFrame(graph) {
  setWeights(weights);
}

void ExecutionFrame::setWeights(const Weights& weights) {
  weightVersion_ = weights.version();

  const auto& inputsToWeights = graph_.signature().inputsToWeights();
  for (const auto& [inputName, weightName] : inputsToWeights) {
    const Value* value = graph_.getValue(inputName);
    setPersistentIValue(value->id(), weights.at(weightName));
  }

  const auto& inputsToCustomObjs = graph_.signature().inputsToCustomObjs();
  for (const auto& [inputName, customObjName] : inputsToCustomObjs) {
    const Value* value = graph_.getValue(inputName);
    setPersistentIValue(value->id(), weights.getCustomObj(customObjName));
  }

  for (const auto& [value, tensor] : weights.getFoldedConsts()) {
    setPersistentIValue(foldedConstIds_.at(value), tensor);
  }

  for (const auto& [v, iv] : weights.getConstFoldedValues()) {
    setPersistentIValue(v, iv);
  }
}

ExecutionFrame::ExecutionFrame(
    const Graph& graph,
    size_t numValues,
    const std::vector<ValueId>&,
    const std::vector<ValueId>&)
    : graph_(graph) {
  allValues_.resize(numValues);
}

void ExecutionFrame::setIValue(ValueId id, c10::IValue ivalue) {
  DCHECK(id < allValues_.size());
  allValues_[id] = std::move(ivalue);
}

at::Tensor ExecutionFrame::getTensor(ValueId id) const {
  const auto& ivalue = getIValue(id);
  if (C10_LIKELY(ivalue.isTensor())) {
    return ivalue.toTensor();
  } else {
    throw std::runtime_error("getTensor called on non-tensor value");
  }
}

std::vector<c10::IValue> ExecutionFrame::getUserOutputs() const {
  std::vector<c10::IValue> ret;
  ret.reserve(graph_.userOutputs().size());
  for (const auto& outputValue : graph_.userOutputs()) {
    if (std::holds_alternative<Value*>(outputValue)) {
      Value* valuePtr = std::get<Value*>(outputValue);
      if (valuePtr) {
        const auto& id = valuePtr->id();
        ret.push_back(getIValue(id));
      }
    } else if (std::holds_alternative<Constant>(outputValue)) {
      const Constant& constValue = std::get<Constant>(outputValue);
      ret.push_back(constantToIValue(constValue));
    }
  }
  return ret;
}

c10::List<c10::IValue> ExecutionFrame::getUserOutputsAsTensorList() const {
  c10::List<c10::IValue> ret(c10::TensorType::get());
  ret.reserve(graph_.userOutputs().size());
  for (const auto& outputValue : graph_.userOutputs()) {
    if (std::holds_alternative<Value*>(outputValue)) {
      Value* valuePtr = std::get<Value*>(outputValue);
      if (valuePtr) {
        const auto& id = valuePtr->id();
        ret.push_back(getIValue(id));
      }
    } else if (std::holds_alternative<Constant>(outputValue)) {
      const Constant& constValue = std::get<Constant>(outputValue);
      ret.push_back(constantToIValue(constValue));
    }
  }
  return ret;
}

std::unordered_map<std::string, at::Tensor> ExecutionFrame::getAllOutputs()
    const {
  std::unordered_map<std::string, at::Tensor> ret;
  for (const auto& outputValue : graph_.outputs()) {
    const auto& name = outputValue->name();
    const auto& id = outputValue->id();
    ret.emplace(name, getTensor(id));
  }
  return ret;
}

std::unordered_map<std::string, at::Tensor> ExecutionFrame::getBufferMutations()
    const {
  // key is buffer name, value is tensor to be written to buffer
  std::unordered_map<std::string, at::Tensor> ret;
  const auto& buffersToMutate = graph_.signature().buffersToMutate();
  for (auto& [mutationOutputName, bufferName] : buffersToMutate) {
    const auto& id = graph_.getValue(mutationOutputName)->id();
    ret.emplace(bufferName, getTensor(id));
  }
  return ret;
}

} // namespace torch::nativert
