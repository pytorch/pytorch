#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

using value_map = std::unordered_map<Value*, Value*>;
using value_set = std::unordered_set<Value*>;

Value* addAndPutAfter(Value *a, Value *b, Node *node) {
  Graph *graph = node->owningGraph();
  Node *add_node = graph->create(kadd, {a, b})
                        ->t_(kalpha, at::Scalar(1).toTensor());
  add_node->insertAfter(node);
  Value *add_output = add_node->output();
  add_output->setType(a->typeOption());
  return add_output;
}


std::unordered_set<Symbol> differentiable_kinds = {
  kadd, ksub, kmul,
};

bool isDifferentiable(Node * n) {
  return differentiable_kinds.count(n->kind()) > 0;
}

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

// Before:
//   - graph has only stage 0
//   - grad_desc doesn't have any fields initialized
// After:
//   - graph has stage 0 and stage 1 that computes its vjp
//   - grad_desc has df_input_vjps and df_output_vjps set
//     (but df_input_vjps will be modified later as well)
static value_map addReverseInline(Graph& graph, Gradient& grad_desc) {
  JIT_ASSERT(graph.stage() == 0);
  graph.advanceStage();

  value_map grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) { return grad_map.at(v); };
  const auto set_grad = [&](Value *x, Value *dx) {
    if (Value * prev_grad = grad_map[x]) {
      Value * new_grad = addAndPutAfter(prev_grad, dx, dx->node());
      grad_map[x] = new_grad;
    } else {
      grad_map[x] = dx;
    }
  };

  auto outputs = graph.outputs();
  for (std::size_t i = 0, num_outputs = outputs.size(); i < num_outputs; ++i) {
    Value * output = outputs[i];
    Value * output_grad = graph.addInput()->setType(output->typeOption());
    set_grad(output, output_grad);
    grad_desc.df_input_vjps.push_back(i);
  }

  for (auto it = graph.rbegin(), end = graph.rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    value_list grad_inputs = gradientForNode(node, fmap(node->outputs(), get_grad));
    JIT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (std::size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      set_grad(inputs[i], grad_inputs[i]);
    }
  }

  auto inputs = graph.inputs();
  for (std::size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    Value * input = inputs[i];
    if (input->stage() > 0) break;
    graph.registerOutput(get_grad(input));
    grad_desc.df_output_vjps.push_back(i);
  }

  return grad_map;
}

// This function will take the graph and return a new one that:
//   - contains all nodes of graph that have given stage
//   - there will be an input corresponding to each input of the inputs array
//   - values corresponding to outputs will be returned from the new graph
// It requires that values contained in inputs are sufficient to be able to
// compute all values in a given stage. An exception will be thrown if this is
// not the case.
static std::shared_ptr<Graph> splitOffStage(
        Graph& graph,
        std::size_t stage,
        ArrayRef<Value*> inputs,
        ArrayRef<Value*> outputs) {
  auto graph_clone = std::make_shared<Graph>();

  value_map val_map; // values in graph -> values in graph_clone
  const auto lookup_val = [&](Value *v) { return val_map.at(v); };

  for (Value *input : inputs)
    val_map[input] = graph_clone->addInput()->setType(input->typeOption());

  for (Node *node : graph.nodes()) {
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

// Takes a graph returned from `addReverseInline` and splits it into two graphs
// (one for each stage). All intermediates needed in the second stage are added to
// outputs of the first graph, and taken as inputs in the second one. For a more
// detailed description see Note [Gradient graphs] in autodiff.h.
// This function also initializes the fields in grad_desc that were undefined after
// `addReverseInline` (and modifies `df_input_vjps`).
static void lambdaLiftReverse(Graph& graph, value_map& grad_map, Gradient& grad_desc) {
  static const auto is_stage_0 = [](Value *v) { return v->stage() == 0; };
  static const auto is_stage_1 = [](Value *v) { return v->stage() == 1; };
  // NOTE: in the comments inside this function first stage is stage 0
  JIT_ASSERT(graph.stage() == 1);

  // --------------------------------------------------------------------------
  // 1. Find values of stage 0 that need to be captured.
  // --------------------------------------------------------------------------
  // First, we need to find all values that are produced in the first stage,
  // and used in the second one. They will need to be added as inputs of the reverse
  // graph, and some of them may also need to be appended as outputs of the primal graph.
  value_set reverse_captures_set;
  value_list reverse_captures; // Invariant: topo sorted
  auto check_uses = [&](Value *v) {
    for (auto use : v->uses()) {
      if (use.user->stage() != 1) continue;
      if (/* bool unseen = */ reverse_captures_set.emplace(v).second) {
        reverse_captures.push_back(v);
      }
    }
  };
  for (Value * input : graph.inputs()) {
    if (input->stage() != 0) break;
    check_uses(input);
  }
  for (Node * node : graph.nodes()) {
    if (node->stage() != 0) break;
    for (Value * output : node->outputs())
      check_uses(output);
  }

  // --------------------------------------------------------------------------
  // 2. Prepare input/outputs lists for both graphs.
  // --------------------------------------------------------------------------
  // It's simple to construct primal_inputs/reverse_outputs,
  // but primal_outputs/reverse_inputs are much more subtle.
  // Here's a summary of how they are supposed to look like:
  //
  // Primal outputs:
  //   [original outputs], [temporaries]
  //
  // Reverse inputs:
  //   [captured primal values, in topological order],
  //   [output vjps (aka grad_outputs)], [temporary vjps]

  // -- Simple cases -----------------------------------------------------------
  value_list primal_inputs   = filter(graph.inputs(),  is_stage_0);
  value_list reverse_outputs = filter(graph.outputs(), is_stage_1);

  // -- Construct primal_outputs, df_input_captures, f_real_outputs ----
  value_list primal_outputs  = filter(graph.outputs(), is_stage_0);
  grad_desc.f_real_outputs = primal_outputs.size();

  std::unordered_map<Value*, std::size_t> orig_primal_outputs_idx;
  std::unordered_map<Value*, std::size_t> orig_primal_inputs_idx;
  // NOTE: we use emplace to avoid replacing an existing index if an output is repeated
  for (std::size_t i = 0, num_outputs = primal_outputs.size(); i < num_outputs; ++i)
    orig_primal_outputs_idx.emplace(primal_outputs[i], i);
  for (std::size_t i = 0, num_inputs = primal_inputs.size(); i < num_inputs; ++i)
    orig_primal_inputs_idx[primal_inputs[i]] = i;

  // NB: reverse_captures are already deduplicated, and in topo order
  for (Value * capture_val : reverse_captures) {
    // If it's already an output we don't have to add anything,
    // but register the fact that it needs to be captured.
    if (orig_primal_outputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captures.emplace_back(Capture::Kind::Output,
                                              orig_primal_outputs_idx[capture_val]);
    // If it's an input, we could add it as an output but in fact it's
    // more efficient to use a special kind of capture.
    } else if (orig_primal_inputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captures.emplace_back(Capture::Kind::Input,
                                               orig_primal_inputs_idx.at(capture_val));
    // Otherwise it's just a regular intermediate value that we need to add as an output
    } else {
      primal_outputs.emplace_back(capture_val);
      grad_desc.df_input_captures.emplace_back(Capture::Kind::Output,
                                               primal_outputs.size() - 1);
    }
  }

  // -- Add VJPs for temporaries, adjust df_input_vjps -------------------------
  // NB [possible optimization]: use the newly added vjp input as soon as the first
  // vjp for that value is generated, to reduce the lifespan of this input
  // (currently we add it to the final vjp after all adds).
  JIT_ASSERT(graph.stage() == 1); // We will be adding inputs to stage 1
  for (std::size_t i = grad_desc.f_real_outputs; i < primal_outputs.size(); ++i) {
    Value * tmp = primal_outputs.at(i);
    Value * tmp_vjp_in = graph.addInput()->setType(tmp->typeOption());
    if (grad_map.count(tmp) == 0) continue; // This gradient wasn't even used.
    Value * tmp_vjp_prev = grad_map.at(tmp);
    // This is quite weird because we can't first make a sum and then replace all uses
    // of tmp_vjp_prev (that would replace its use in the sum too!), so we create an
    // incorrect sum that doesn't use prev vjp, replace uses, and fix the sum.
    Value * new_vjp = addAndPutAfter(tmp_vjp_in, tmp_vjp_in, tmp_vjp_prev->node());
    tmp_vjp_prev->replaceAllUsesWith(new_vjp);
    new_vjp->node()->replaceInput(1, tmp_vjp_prev);
    grad_desc.df_input_vjps.emplace_back(i);
  }

  // -- Construct reverse_inputs -----------------------------------------------
  // Quick reference:
  //   [captured primal values, in topological order],            1st loop below
  //   [output vjps (aka grad_outputs)], [temporary vjps]         2nd loop below
  value_list reverse_inputs;
  for (Capture capture : grad_desc.df_input_captures) {
    auto & source = capture.kind == Capture::Kind::Input ? primal_inputs : primal_outputs;
    reverse_inputs.push_back(source[capture.offset]);
  }
  // These are the vjps computed by differentiate + the code above
  for (Value * reverse_vjp : filter(graph.inputs(), is_stage_1))
    reverse_inputs.push_back(reverse_vjp);

  // Finally, we can split the graph into two parts.
  grad_desc.f  = splitOffStage(graph, 0, primal_inputs, primal_outputs);
  grad_desc.df = splitOffStage(graph, 1, reverse_inputs, reverse_outputs);
}

Gradient differentiate(std::shared_ptr<Graph>& _graph) {
  // Take ownership of the graph
  std::shared_ptr<Graph> graph;
  JIT_ASSERTM(_graph.use_count() == 1,
              "differentiate will mutate and destroy the graph, so it requires "
              "graph.use_count() == 1");
  std::swap(_graph, graph);
  // XXX: Take care when handling outputs - they can be duplicated!
  Gradient grad_desc;
  // Fills in df_input_vjps and df_output_vjps
  auto grad_map = addReverseInline(*graph, grad_desc);
  // Fills in f, df, f_real_outputs, df_input_captures,
  // modifies df_input_vjps (new vjps are added for temporaries)
  lambdaLiftReverse(*graph, grad_map, grad_desc);
  return grad_desc;
}

}}
