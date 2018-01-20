#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

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

static std::shared_ptr<Graph> splitOffStage(
        const std::shared_ptr<Graph>& graph,
        std::size_t stage,
        ArrayRef<Value*> inputs,
        ArrayRef<Value*> outputs) {
  auto graph_clone = std::make_shared<Graph>();

  std::unordered_map<Value*, Value*> val_map; // values in graph -> values in graph_clone
  const auto lookup_val = [&](Value *v) { return val_map.at(v); };

  for (Value *input : inputs)
    val_map[input] = graph_clone->addInput()->setType(input->typeOption());

  for (Node *node : *graph) {
    if (node->stage() != stage) continue;
    Node *node_clone = graph_clone->createClone(node, lookup_val);
    for (std::size_t i = 0, num_outputs = node_clone->outputs().size(); i < num_outputs; ++i)
      val_map[node->outputs()[i]] = node_clone->outputs()[i];
    graph_clone->appendNode(node_clone);
  }

  for (Value *output : outputs) {
    JIT_ASSERT(output->stage() == stage);
    graph_clone->registerOutput(val_map.at(output));
  }

  return graph_clone;
}

static std::unordered_map<Value*, int64_t> computeValueTopoOrder(const std::shared_ptr<Graph>& graph) {
  int64_t next_nr = 0;
  std::size_t stage = 0;
  std::unordered_map<Value*, int64_t> order;
  auto inputs_it = graph->inputs().begin();
  auto inputs_end = graph->inputs().end();
  auto nodes_it = graph->nodes().begin();
  auto nodes_end = graph->nodes().end();

  while (inputs_it != inputs_end || nodes_it != nodes_end) {
    for (; inputs_it != inputs_end; ++inputs_it) {
      Value *input = *inputs_it;
      if (input->stage() != stage) break;
      order[input] = next_nr++;
    }
    for (; nodes_it != nodes_end; ++nodes_it) {
      Node *node = *nodes_it;
      if (node->stage() != stage) break;
      for (Value * output : node->outputs())
        order[output] = next_nr++;
    }
    // NOTE: outputs are either inputs or are produced by nodes,
    // so we don't need to handle them in any way.
    stage++;
  }
  JIT_ASSERT(stage == graph->stage() + 1);
  return order;
}

LiftedReverse lambdaLiftReverse(std::shared_ptr<Graph>& graph) {
  static const auto is_stage_0 = [](Value *v) { return v->stage() == 0; };
  static const auto is_stage_1 = [](Value *v) { return v->stage() == 1; };
  // NOTE: in the comments here first stage == stage 0
  JIT_ASSERT(graph->stage() == 1);

  // --------------------------------------------------------------------------
  // 1. Find values of stage 0 that need to ba captured.
  // --------------------------------------------------------------------------
  // First, we need to find all values that are produced in the first stage,
  // and used in the second one. They will need to be added as inputs of the reverse
  // graph, and some of them may also need to be appended as outputs of the primal graph.
  std::unordered_set<Value*> extra_reverse_inputs;
  for (Node * node : *graph) {
    if (node->stage() == 0) continue;
    for (Value * input : node->inputs()) {
      if (input->stage() != 0) continue;
      extra_reverse_inputs.insert(input);
    }
  }

  // --------------------------------------------------------------------------
  // 2. Establish an ordering on inputs/outputs.
  // --------------------------------------------------------------------------
  // We will need to sort primal outputs and reverse inputs, so that they maintain
  // an order of:
  //   [primal outputs], [primal temporaries], [reverse inputs]
  // where each section is sorted in the same way as in the input graph.
  //
  // This corresponds exactly to taking a topological ordering on values,
  // but artificially boosting outputs of the first stage to precede all
  // other values.
  auto topo_order = computeValueTopoOrder(graph);
  auto io_compare = [&](Value *a, Value *b) { return topo_order.at(a) < topo_order.at(b); };
  { // Boost outputs of primal graph
    int64_t next_nr = -static_cast<int64_t>(graph->outputs().size());
    for (Value * output : graph->outputs()) {
      if (output->stage() != 0) break;
      topo_order[output] = next_nr++;
    }
  }

  // --------------------------------------------------------------------------
  // 3. Prepare input/outputs lists for both graphs.
  // --------------------------------------------------------------------------
  // It's simple to construct primal_inputs/reverse_outputs,
  // but primal_outputs/reverse_inputs are much more subtle.
  // Here's a summary of how they are supposed to look like:
  //
  // Primal outputs:
  //   [original outputs], [temporaries]
  //
  // Reverse inputs:
  //   [original outputs#], [temporaries], [output vjps (aka grad_outputs)]
  //
  // # this is an arbitrary subset of original outputs, that maintains the original ordering.
  //   IMPORTANT: THIS MEANS THAT YOU CAN'T JUST TAKE OUTPUTS OF PRIMAL GRAPH AND FEED THEM
  //              AS THE INPUTS OF THE REVERSE GRAPH.

  // XXX: Take care when handling primal_outputs here - you can't put them in a set and
  // just take them from it, because some values could have appeared multiple times in the
  // outputs list, and this will deduplicate them.

  value_list primal_inputs   = filter(graph->inputs(),  is_stage_0);
  value_list primal_outputs  = filter(graph->outputs(), is_stage_0);
  value_list reverse_inputs  = filter(graph->inputs(),  is_stage_1);
  value_list reverse_outputs = filter(graph->outputs(), is_stage_1);

  // Cache some information that will be useful in stage 4.
  auto num_original_primal_outputs = primal_outputs.size();
  auto num_original_reverse_inputs = reverse_inputs.size();

  // Add extra inputs to the reverse graph
  reverse_inputs.insert(reverse_inputs.begin(), extra_reverse_inputs.begin(),
                                                extra_reverse_inputs.end());
  std::sort(reverse_inputs.begin(), reverse_inputs.end(), io_compare);

  // We will mutate the set in a moment, so it will no longer contain all new inputs
  auto new_primal_outputs = std::move(extra_reverse_inputs);
  // Remove outputs that already appear in the list -
  // there's no need to return them multiple times.
  for (Value * output : primal_outputs) new_primal_outputs.erase(output);

  // Add extra outputs to the primal graph
  primal_outputs.insert(primal_outputs.end(), new_primal_outputs.begin(),
                                              new_primal_outputs.end());
  std::sort(primal_outputs.begin(), primal_outputs.end(), io_compare);

  // --------------------------------------------------------------------------
  // 4. Split graphs and compute metadata
  // --------------------------------------------------------------------------
  // We're done. Split off the graphs and compute metadata about the relation of
  // inputs/outputs of these graphs.
  uint64_t next_idx = 0;
  std::unordered_map<Value*, uint64_t> f_output_idx;
  for (Value * output : primal_outputs)
    f_output_idx[output] = next_idx++;
  auto num_captured_rev_inputs = reverse_inputs.size() - num_original_reverse_inputs;
  auto captured_rev_inputs = ArrayRef<Value*>(reverse_inputs).slice(0, num_captured_rev_inputs);

  LiftedReverse res;
  res.f  = splitOffStage(graph, 0, primal_inputs, primal_outputs);
  res.df = splitOffStage(graph, 1, reverse_inputs, reverse_outputs);
  res.df_input_captures = fmap(captured_rev_inputs, [&](Value* v) { return f_output_idx.at(v); });
  res.f_output_intermediates = primal_outputs.size() - num_original_primal_outputs;
  return res;
}

}}
