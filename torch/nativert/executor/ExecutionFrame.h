#pragma once

#include <unordered_map>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/Graph.h>

#include <c10/util/Logging.h>

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

  ~ExecutionFrame() {
    destroyBorrowedIValues();
  }

  std::vector<c10::IValue> tryMoveUserOutputs();

  c10::IValue moveIValue(ValueId id) {
    return std::move(allValues_[id]);
  }

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
  void setBorrowedIValue(ValueId id, c10::IValue ivalue);

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

  const std::vector<bool>& persistentValues() const {
    return persistent_;
  }

  void setPersistentIValue(ValueId id, c10::IValue ivalue) {
    setIValue(id, std::move(ivalue));
    persistent_[id] = true;
  }

  void releaseValue(ValueId id) {
    CHECK(!persistent_[id]) << "Cannot release persistent value";
    allValues_[id] = c10::IValue();
  }

  void destroyBorrowedIValues() {
    for (const auto& id : borrowedValueIds_) {
      c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(getIValue(id));
    }
    borrowedValueIds_.clear();
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
  bool isOutputMovable(size_t idx) const {
    TORCH_CHECK_LT(idx, moveable_output_mask_.size());
    return moveable_output_mask_[idx];
  }
  void updateMovableOutputs();

  const Graph& graph_;
  WeightVersion weightVersion_ = -1;

  // All the intermediate values for the entire graph, including graph inputs
  // and outputs This table is fixed once constructed
  std::vector<c10::IValue> allValues_;
  std::vector<bool> persistent_;

  std::unordered_map<int64_t, c10::intrusive_ptr<c10d::Work>> work_;

  std::vector<ValueId> borrowedValueIds_;

  std::unordered_map<std::string, ValueId> foldedConstIds_;

  // moveable_output_mask_[i] corresponds to user_outputs_[i]
  //
  // if moveable_output_mask_[i] is true, then user_outputs_[i]
  // can be moved
  std::vector<bool> moveable_output_mask_;
};

} // namespace torch::nativert
