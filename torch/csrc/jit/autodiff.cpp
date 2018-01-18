#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

using value_list = std::vector<Value*>;

static std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  const auto build_sym_grad = [node](const std::vector<SymbolicVariable>& grads) -> std::vector<SymbolicVariable> {
    auto inputs = node->inputs();
    switch(node->kind()) {
      case kadd:
        return {grads[0], grads[0]};
      case ksub:
        return {grads[0], -grads[0]};
      case kmul:
        return {grads[0] * inputs[1], grads[0] * inputs[0]};
    }
    throw std::runtime_error(std::string("don't support differentiation of `") +
                            node->kind().toString() + "`");
  };
  auto sym_grads = build_sym_grad(fmap<SymbolicVariable>(grad_values));
  return fmap(sym_grads, [](const SymbolicVariable &v) { return v.value(); });
}

void differentiate(std::shared_ptr<Graph>& graph) {
  JIT_ASSERT(graph->stage() == 0);
  graph->advanceStage();

  std::unordered_map<Value*, Value*> grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) { return grad_map[v]; };
  for (auto output : graph->outputs())
    grad_map[output] = graph->addInput()->setType(output->typeOption());

  for (auto it = graph->rbegin(), end = graph->rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    value_list grad_inputs = gradientForNode(node, fmap(node->outputs(), get_grad));
    JIT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (std::size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      if (Value * prev_grad = grad_map[inputs[i]]) {
        Node *new_grad_node = graph->create(kadd, {prev_grad, grad_inputs[i]})
                                   ->t_(kalpha, at::Scalar(1).toTensor());
        new_grad_node->insertAfter(grad_inputs[i]->node());
        Value *new_grad = new_grad_node->output();
        new_grad->setType(prev_grad->typeOption());
        grad_map[inputs[i]] = new_grad;
      } else {
        grad_map[inputs[i]] = grad_inputs[i];
      }
    }
  }

  for (auto input : graph->inputs()) {
    if (input->stage() > 0) break;
    graph->registerOutput(grad_map.at(input));
  }
}

}}
