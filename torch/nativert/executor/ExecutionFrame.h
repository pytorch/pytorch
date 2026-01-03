#pragma once

#include <unordered_map>

#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/executor/memory/LayoutManager.h>
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

  explicit ExecutionFrame(
      const Graph& graph,
      const Weights& weights,
      const torch::nativert::ExecutorConfig& executorConfig = {},
      LayoutPlanner* layoutPlanner = nullptr);

  // Constructor for testing purpose
  explicit ExecutionFrame(
      const Graph& graph,
      size_t numValues,
      const std::vector<ValueId>& graphInputIds,
      const std::vector<ValueId>& graphOutputIds);

  ExecutionFrame(const ExecutionFrame&) = delete;
  ExecutionFrame& operator=(const ExecutionFrame&) = delete;
  ExecutionFrame(ExecutionFrame&&) = delete;
  ExecutionFrame& operator=(ExecutionFrame&&) = delete;

  ~ExecutionFrame() {
    destroyBorrowedIValues();
  }

  template <typename CB>
  auto withManagedMemory(CB&& cb) {
    if (!layoutManager_) {
      return std::forward<CB>(cb)(nullptr);
    }

    LayoutManagerGuard guard(*layoutManager_);
    return std::forward<CB>(cb)(
        const_cast<const LayoutManager*>(layoutManager_.get()));
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

  C10_ALWAYS_INLINE bool isManagedValue(const ValueId id) const {
    return layoutPlanner_ != nullptr && layoutPlanner_->is_managed(id);
  }

  void setPersistentIValue(ValueId id, c10::IValue ivalue) {
    setIValue(id, std::move(ivalue));
    persistent_[id] = true;
  }

  void releaseValueIfNeeded(ValueId id) {
    if (!isManagedValue(id) && !persistent_[id]) {
      allValues_[id] = c10::IValue();
    }
  }

  void destroyBorrowedIValues() {
    for (const auto& id : borrowedValueIds_) {
      c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(getIValue(id));
    }
    borrowedValueIds_.clear();
  }

  WeightVersion weightVersion() const {
    return weightVersion_;
  }

  void setWeights(const Weights& weights);

  static std::vector<std::pair<ValueId, c10::IValue>> getPersistentValues(
      const Graph& graph,
      const Weights* weights = nullptr);

  static std::vector<bool> getPersistentValueMask(
      const Graph& graph,
      const Weights* weights = nullptr) {
    std::vector<bool> persistentValuesMask(graph.numValues());
    for (auto& [valueId, _] : getPersistentValues(graph, weights)) {
      persistentValuesMask[valueId] = true;
    }
    return persistentValuesMask;
  }

 private:
  bool isOutputMovable(size_t idx) const {
    TORCH_CHECK(idx < moveable_output_mask_.size());
    return moveable_output_mask_[idx];
  }

  void updatePersistentValues(const Weights* weights = nullptr);
  void updateMovableOutputs();

  const Graph& graph_;
  WeightVersion weightVersion_ = -1;

  std::unique_ptr<LayoutManager> layoutManager_;
  LayoutPlanner* layoutPlanner_{nullptr};

  // All the intermediate values for the entire graph, including graph inputs
  // and outputs This table is fixed once constructed
  std::vector<c10::IValue> allValues_;
  // a class-local version of getPersistentValueMask
  std::vector<bool> persistent_;

  std::vector<ValueId> borrowedValueIds_;

  // moveable_output_mask_[i] corresponds to user_outputs_[i]
  //
  // if moveable_output_mask_[i] is true, then user_outputs_[i]
  // can be moved
  std::vector<bool> moveable_output_mask_;
};

} // namespace torch::nativert
