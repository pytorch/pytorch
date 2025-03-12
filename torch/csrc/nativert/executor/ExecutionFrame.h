#pragma once

#include <unordered_map>

#include "torch/csrc/nativert/executor/Weights.h"
#include "torch/csrc/nativert/graph/Graph.h"

#include <c10/util/Logging.h>

#include <torch/csrc/distributed/c10d/Work.hpp> // @manual

namespace torch::nativert {

/**
 * This class encapsulate the stateful values of an execution,
 * most notably, the tensor values passed between nodes, aka intermediate
 * activations.
 */
class ExecutionFrame {
 public:
  // Constructor for weight-less graph, used for higher order ops, e.g.
  // torch.cond
  explicit ExecutionFrame(const Graph& graph);

  explicit ExecutionFrame(const Graph& graph, const Weights& weights);

  // Constructor for testing purpose
  explicit ExecutionFrame(
      const Graph& graph,
      size_t numValues,
      const std::vector<ValueId>& graphInputIds,
      const std::vector<ValueId>& graphOutputIds);

  ~ExecutionFrame() {}

  std::vector<c10::IValue> getUserOutputs() const;
  c10::List<c10::IValue> getUserOutputsAsTensorList() const;

  std::unordered_map<std::string, at::Tensor> getBufferMutations() const;

  std::unordered_map<std::string, at::Tensor> getAllOutputs() const;

  const c10::IValue& getIValue(ValueId id, bool allowNone = true) const {
    const auto& iValue = allValues_[id];
    if (allowNone && iValue.isNone()) {
      return iValue;
    }
    DCHECK(!iValue.isNone());
    return iValue;
  }

  c10::IValue& getIValue(ValueId id, bool allowNone = true) {
    auto& iValue = allValues_[id];
    if (allowNone && iValue.isNone()) {
      return iValue;
    }
    DCHECK(!iValue.isNone());
    return iValue;
  }

  void setIValue(ValueId id, c10::IValue ivalue);

  at::Tensor getTensor(ValueId id) const;

  std::vector<at::Tensor> getTensorVector(ValueId id) const {
    return getIValue(id).toTensorVector();
  }

  int64_t getSymInt(ValueId id) const {
    return getIValue(id).toInt();
  }

  double getSymFloat(ValueId id) const {
    return getIValue(id).toDouble();
  }

  void setPersistentIValue(ValueId id, c10::IValue ivalue) {
    setIValue(id, std::move(ivalue));
    persistent_[id] = true;
  }

  void releaseValue(ValueId id) {
    CHECK(!persistent_[id]) << "Cannot release persistent value";
    allValues_[id] = c10::IValue();
  }

  void releaseUserOutputs() {
    for (const auto& outputValue : graph_.userOutputs()) {
      if (std::holds_alternative<Value*>(outputValue)) {
        Value* valuePtr = std::get<Value*>(outputValue);
        if (valuePtr) {
          const auto& id = valuePtr->id();
          if (!persistent_[id]) {
            releaseValue(id);
          }
        }
      }
    }
  }

  void setWork(int64_t workId, const c10::intrusive_ptr<c10d::Work>& work) {
    work_[workId] = work;
  }

  c10::intrusive_ptr<c10d::Work> getWork(int64_t workId) const {
    CHECK(work_.find(workId) != work_.end())
        << "Couldn't find work with Id: " << workId;
    return work_.at(workId);
  }

  WeightVersion weightVersion() const {
    return weightVersion_;
  }

  void setWeights(const Weights& weights);

 private:
  const Graph& graph_;
  WeightVersion weightVersion_ = -1;

  // All the intermediate values for the entire graph, including graph inputs
  // and outputs This table is fixed once constructed
  std::vector<c10::IValue> allValues_;
  std::vector<bool> persistent_;

  std::unordered_map<int64_t, c10::intrusive_ptr<c10d::Work>> work_;

  std::unordered_map<std::string, ValueId> foldedConstIds_;
};

} // namespace torch::nativert
