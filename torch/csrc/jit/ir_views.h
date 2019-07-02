#include <torch/csrc/jit/ir.h>

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

 private:
  Node* node_;

  // adjust index_ordering by adding indices 0 - thorugh adjust, and
  // incrementing all existing inputs by adjust
  static std::vector<size_t> adjustIndices(
      size_t adjust,
      const std::vector<size_t>& index_ordering) {
    std::vector<size_t> adjusted;
    adjusted.reserve(adjust + index_ordering.size());
    for (size_t i = 0; i < adjust; ++i) {
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
