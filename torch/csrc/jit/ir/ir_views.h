#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct IfView {
  explicit IfView(Node* node) : node_(node) {
    AT_ASSERT(node->kind() == ::c10::prim::If);
  }
  Value* cond() const {
    return node_->input(0);
  }
  Block* thenBlock() const {
    return node_->blocks().at(0);
  }
  Block* elseBlock() const {
    return node_->blocks().at(1);
  }
  ArrayRef<Value*> thenOutputs() const {
    return thenBlock()->outputs();
  }
  ArrayRef<Value*> elseOutputs() const {
    return elseBlock()->outputs();
  }
  ArrayRef<Value*> outputs() const {
    return node_->outputs();
  }
  Node* node() const {
    return node_;
  }
  operator Node*() const {
    return node_;
  }

  void permuteOutputs(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    thenBlock()->permuteOutputs(new_output_order);
    elseBlock()->permuteOutputs(new_output_order);
  }

 private:
  Node* node_;
};

struct LoopView {
  explicit LoopView(Node* node) : node_(node) {
    AT_ASSERT(
        node->kind() == ::c10::prim::Loop || node->kind() == ::c10::onnx::Loop);
  }
  Block* bodyBlock() const {
    return node_->blocks().at(0);
  }
  Value* cond() const {
    return node_->input(0);
  }
  Value* maxTripCount() const {
    return node_->input(0);
  }
  Value* inputCond() const {
    return node_->input(1);
  }
  Value* nextCond() const {
    return bodyBlock()->outputs().at(0);
  }
  Value* currentTripCount() const {
    return bodyBlock()->inputs().at(0);
  }
  ArrayRef<Value*> carriedInputs() const {
    // skip trip count and cond
    return node_->inputs().slice(2);
  }
  ArrayRef<Value*> carriedInputsWithCond() const {
    // skip trip count and cond
    return node_->inputs().slice(1);
  }
  ArrayRef<Value*> carriedOutputs() const {
    return node_->outputs();
  }
  ArrayRef<Value*> bodyCarriedInputs() const {
    // skip trip count and cond
    return bodyBlock()->inputs().slice(1);
  }
  ArrayRef<Value*> bodyCarriedOutputs() const {
    return bodyBlock()->outputs().slice(1);
  }
  Node* node() const {
    return node_;
  }
  operator Node*() const {
    return node_;
  }

  void permuteLoopCarried(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    // skip trip count and cond
    node_->permuteInputs(adjustIndices(2, new_output_order));
    auto adjusted_block_order = adjustIndices(1, new_output_order);
    bodyBlock()->permuteOutputs(adjusted_block_order);
    bodyBlock()->permuteInputs(adjusted_block_order);
  }

  void replaceMaxTripCount(Value* new_max_trip_count) {
    node_->replaceInput(0, new_max_trip_count);
  }
  void replaceInputCondition(Value* new_input_condition) {
    node_->replaceInput(1, new_input_condition);
  }

  // our way of encoding loops makes them difficult to turn back into python
  // syntax. we have to check properties of the condition and trip count inputs
  // to figure out which one it initially was. ModifiedLoops are not directly
  // mappable to either For or While
  enum LoopType { While, For, ModifiedLoop };

  LoopType loopType() {
    auto trip_count = toIValue(maxTripCount());
    auto cond_input = toIValue(inputCond());
    auto cond_next = toIValue(nextCond());

    bool condition_is_always_true =
        cond_input && cond_input->toBool() && cond_next && cond_next->toBool();
    bool trip_count_is_specified = !trip_count || // trip is not a constant
        trip_count->toInt() !=
            std::numeric_limits<int64_t>::max() || // it is a constant but not
                                                   // the default one
        currentTripCount()->uses().size() >
            0; // it is actually being used in the body.

    if (condition_is_always_true) {
      // if the trip count was not specified this was a user-written while True:
      return trip_count_is_specified ? For : While;
    } else {
      if (trip_count_is_specified) {
        return ModifiedLoop;
      }
      return While;
    }
  }

 private:
  Node* node_;

  // adjust index_ordering by adding indices 0 - thorugh adjust, and
  // incrementing all existing inputs by adjust
  static std::vector<size_t> adjustIndices(
      size_t adjust,
      const std::vector<size_t>& index_ordering) {
    std::vector<size_t> adjusted;
    adjusted.reserve(adjust + index_ordering.size());
    for (const auto i : c10::irange(adjust)) {
      adjusted.push_back(i);
    }
    for (auto index : index_ordering) {
      adjusted.push_back(index + adjust);
    }
    return adjusted;
  }
};
} // namespace jit
} // namespace torch
