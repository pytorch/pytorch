#include <c10/util/Enumerate.h>
#include <c10/util/Logging.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/ExecutionPlanner.h>

namespace torch::nativert {

ExecutionFrame::ExecutionFrame(const Graph& graph)
    : graph_(graph),
      allValues_(graph.numValues()),
      persistent_(graph.numValues()),
      moveable_output_mask_(graph.userOutputs().size()) {
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

ExecutionFrame::ExecutionFrame(
    const Graph& graph,
    const Weights& weights,
    const torch::nativert::ExecutorConfig& cfg,
    LayoutPlanner* layoutPlanner)
    : ExecutionFrame(graph) {
  setWeights(weights);
  if (layoutPlanner != nullptr) {
    layoutPlanner_ = layoutPlanner;
    layoutManager_ = std::make_unique<LayoutManager>(
        *layoutPlanner,
        *this,
        cfg.layoutPlannerSettings.layoutManagerSettings());
  }
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

  for (const auto& [n, iv] : weights.getConstFoldedValues()) {
    const Value* v = graph_.getValue(n);
    setPersistentIValue(v->id(), iv);
  }

  updateMovableOutputs();
}

void ExecutionFrame::updateMovableOutputs() {
  moveable_output_mask_.assign(moveable_output_mask_.size(), true);

  c10::FastSet<ValueId> inputs;
  for (const auto* input : graph_.userInputs()) {
    if (input) {
      inputs.insert(input->id());
    }
  }

  const auto& outputs = graph_.userOutputs();
  const size_t num_outputs = outputs.size();

  c10::FastSet<ValueId> seen;
  for (size_t i = 0; i < num_outputs; i++) {
    auto idx = num_outputs - 1 - i;
    if (const Value* const* valuePtr = std::get_if<Value*>(&outputs[idx]);
        valuePtr && *valuePtr) {
      auto id = (*valuePtr)->id();

      /*
          values are not moveable if:
          1. they are persistent
          2. they are inputs (since inputs are borrowed)
          3. the value will be moved in a later (right-more) output
      */

      if (!seen.insert(id).second || persistent_[id] ||
          inputs.find(id) != inputs.end()) {
        moveable_output_mask_[idx] = false;
      }
    }
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
  DCHECK(static_cast<size_t>(id) < allValues_.size());
  allValues_[id] = std::move(ivalue);
}

void ExecutionFrame::setBorrowedIValue(ValueId id, c10::IValue ivalue) {
  DCHECK(static_cast<size_t>(id) < allValues_.size());
  borrowedValueIds_.push_back(id);
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

std::vector<c10::IValue> ExecutionFrame::tryMoveUserOutputs() {
  std::vector<c10::IValue> ret;
  const auto& outputs = graph_.userOutputs();
  ret.reserve(outputs.size());
  for (const auto& [i, outputValue] : c10::enumerate(outputs)) {
    if (const Value* const* valuePtr = std::get_if<Value*>(&outputValue);
        valuePtr && *valuePtr) {
      ret.push_back(
          isOutputMovable(i) ? moveIValue((*valuePtr)->id())
                             : getIValue((*valuePtr)->id()));
    } else if (Constant const* constant = std::get_if<Constant>(&outputValue)) {
      ret.push_back(constantToIValue(*constant));
    }
  }
  return ret;
}

} // namespace torch::nativert
