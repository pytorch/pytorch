#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <ATen/core/jit_type.h>

#include <vector>

namespace torch {
namespace jit {

namespace {

bool getRequiresGrad(Value* value) {
  return value->requires_grad();
}

void setRequiresGrad(Value* value, bool req_value) {
  if (auto type = value->type()->cast<DimensionedTensorType>()) {
    value->setType(type->withRequiresGrad(req_value));
  }
}

void setRequiresGrad(
    at::ArrayRef<Value*> outputs,
    const std::vector<bool>& values) {
  AT_ASSERT(outputs.size() == values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    setRequiresGrad(outputs[i], values[i]);
  }
}

void setRequiresGrad(Node* node, const std::vector<bool>& values) {
  setRequiresGrad(node->outputs(), values);
}

std::vector<bool> bitwiseOr(std::vector<bool> a, const std::vector<bool>& b) {
  AT_ASSERT(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = a[i] || b[i];
  }
  return a;
}

void PropagateRequiresGradSimpleNode(Node* node) {
  static const OperatorSet comparison_ops = {
      "aten::lt(Tensor self, Tensor other) -> Tensor",
      "aten::le(Tensor self, Tensor other) -> Tensor",
      "aten::gt(Tensor self, Tensor other) -> Tensor",
      "aten::ge(Tensor self, Tensor other) -> Tensor",
      "aten::eq(Tensor self, Tensor other) -> Tensor",
      "aten::ne(Tensor self, Tensor other) -> Tensor",
      "aten::lt(Tensor self, Scalar other) -> Tensor",
      "aten::le(Tensor self, Scalar other) -> Tensor",
      "aten::gt(Tensor self, Scalar other) -> Tensor",
      "aten::ge(Tensor self, Scalar other) -> Tensor",
      "aten::eq(Tensor self, Scalar other) -> Tensor",
      "aten::ne(Tensor self, Scalar other) -> Tensor",
  };

  if (comparison_ops.find(node)) {
    return setRequiresGrad(node->output(), false);
  } else if (node->matches(
                 "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
    return setRequiresGrad(node->output(), node->input(0)->requires_grad());
  } else if (node->matches("aten::detach(Tensor self) -> Tensor")) {
    return setRequiresGrad(node->output(), false);
  }

  auto inputs = node->inputs();
  auto outputs = node->outputs();
  bool should_require =
      std::any_of(inputs.begin(), inputs.end(), getRequiresGrad);
  for (Value* output : outputs) {
    if (auto type = output->type()->cast<DimensionedTensorType>()) {
      setRequiresGrad(
          output, should_require && at::isFloatingType(type->scalarType()));
    }
  }
}

void PropagateRequiresGrad(Block* block);

void PropagateRequiresGrad(Node* node) {
  if (node->kind() == prim::If) {
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    PropagateRequiresGrad(true_block);
    PropagateRequiresGrad(false_block);

    auto outputs_require = bitwiseOr(
        fmap(true_block->outputs(), getRequiresGrad),
        fmap(false_block->outputs(), getRequiresGrad));
    setRequiresGrad(node, outputs_require);
  } else if (node->kind() == prim::Loop) {
    auto body = node->blocks().at(0);
    std::vector<bool> body_inputs_require =
        fmap(node->inputs().slice(2), getRequiresGrad);
    std::vector<bool> body_outputs_require(node->outputs().size(), false);

    while (body_inputs_require != body_outputs_require) {
      body_inputs_require =
          bitwiseOr(body_inputs_require, body_outputs_require);
      setRequiresGrad(
          body->param_node()->outputs().slice(1), body_inputs_require);
      PropagateRequiresGrad(body);
      body_outputs_require =
          fmap(body->return_node()->inputs().slice(1), getRequiresGrad);
    }

    setRequiresGrad(node, body_outputs_require);
  } else {
    PropagateRequiresGradSimpleNode(node);
  }
}

void PropagateRequiresGrad(Block* block) {
  for (Node* node : block->nodes()) {
    PropagateRequiresGrad(node);
  }
}

} // anonymous namespace

void PropagateRequiresGrad(std::shared_ptr<Graph>& graph) {
  PropagateRequiresGrad(graph->block());
}

} // namespace jit
} // namespace torch
