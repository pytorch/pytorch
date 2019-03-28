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

 private:
  Node* node_;
};

} // namespace jit
} // namespace torch
