#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace jit {

using value_map = std::unordered_map<Value*, Value*>;
using value_set = std::unordered_set<Value*>;

// Creates a node for a + b and puts it after the given node.
// If node is a null pointer, appends that node at the end of the node list.
Value* addValues(Value *a, Value *b, Node *node = nullptr) {
  Graph *graph = a->node()->owningGraph();
  Node *add_node = graph->create(kadd, {a, b})
                        ->t_(kalpha, at::Scalar(1).toTensor());
  if (node) {
    add_node->insertAfter(node);
  } else {
    graph->appendNode(add_node);
  }
  Value *add_output = add_node->output();
  add_output->setType(a->typeOption());
  return add_output;
}


std::unordered_set<Symbol> differentiable_kinds = {
  kadd, ksub, kmul, kConstant, kReplaceIfUndef,
};

bool isDifferentiable(Node * n) {
  return differentiable_kinds.count(n->kind()) > 0;
}



static std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  const auto build_sym_grad = [node](const std::vector<SymbolicVariable>& grads) -> std::vector<SymbolicVariable> {
    auto inputs = node->inputs();
    switch(node->kind()) {
      case kadd:
        // o = a - alpha*other
        if(inputs.size() == 1)
          return { grads.at(0) };
          // o = a + alpha*b
        return {grads.at(0), grads.at(0) * at::Scalar(node->t(kalpha)) };
      case ksub:
        // o = a - alpha*other
        if(inputs.size() == 1)
          return {grads.at(0)};
        // o = a - alpha*b
        return {grads.at(0), -grads.at(0) * at::Scalar(node->t(kalpha))};
      case kmul:
        // o = a * other
        if(inputs.size() == 1)
          return {grads.at(0) * at::Scalar(node->t(kother))};
        // o = a * b
        return {grads.at(0) * inputs.at(1), grads.at(0) * inputs.at(0)};
      case kConstant:
        return {};
      case kReplaceIfUndef:
        return {grads.at(0), grads.at(0)};
    }
    throw std::runtime_error(std::string("don't support differentiation of `") +
                            node->kind().toString() + "`");
  };
  auto sym_grads = build_sym_grad(fmap<SymbolicVariable>(grad_values));
  return fmap(sym_grads, [](const SymbolicVariable &v) { return v.value(); });
}

static value_set findAllRequiresGradNodes(
        Graph& graph, const std::vector<bool>& input_requires_grad) {
  JIT_ASSERT(graph.inputs().size() == input_requires_grad.size());

  std::unordered_set<Value*> requires_grad_set;
  const auto requires_grad = [&](Value *v) { return requires_grad_set.count(v) > 0; };

  auto inputs = graph.inputs();
  for (std::size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    if (!input_requires_grad[i]) continue;
    requires_grad_set.emplace(inputs[i]);
  }

  for (Node * node : graph.nodes()) {
    if (std::none_of(node->inputs().begin(), node->inputs().end(), requires_grad)) continue;
    for (Value * output : node->outputs())
      requires_grad_set.emplace(output);
  }

  return requires_grad_set;
}

static Value* createZerosLike(Value *v) {
  JIT_EXPECTM(v->hasType(), "can't allocate zero gradient for a value without a type");
  Graph *graph = v->owningGraph();
  auto type = v->type()->expect<TensorType>();
  AutoGPU gpu_guard(type->device());

  auto & at_type = type->device() == -1 ? at::CPU(type->scalarType()) : at::CUDA(type->scalarType());
  auto zeros = at_type.zeros({1}).expand(type->sizes());
  Node *constant = graph->createConstant(zeros)
                        ->i_(kis_zero, 1);
  graph->appendNode(constant);
  return constant->output();
}

// any vjp input may be undefined, and we need to potentially replace it
// with a zero tensor of the right size if required.
// this function inserts a guard into the graph that does this replacement.
// ReplaceIfUndef(dv,c) replaces dv with c if dv is undef.
// During Graph specialization these guards will get removed when
// 'dv' is known to be undef, and the zeros will be propagated if possible.
static Value* createUndefGuard(Value * dv, Value * alternative) {
  Graph* graph = dv->owningGraph();
  Node * n = graph->create(kReplaceIfUndef, {dv, alternative});
  return graph->appendNode(n)->output();
}

struct ReverseDetails {
  ReverseDetails(value_map&& grad_map, value_set&& requires_grad_set)
    : grad_map(std::move(grad_map))
    , requires_grad_set(std::move(requires_grad_set)) {}

  value_map grad_map;
  value_set requires_grad_set;
};

// Before:
//   - graph has only stage 0
//   - grad_desc doesn't have any fields initialized
// After:
//   - graph has stage 0 and stage 1 that computes its vjp
//   - grad_desc has df_input_vjps and df_output_vjps set
//     (but df_input_vjps will be modified later as well)
static ReverseDetails addReverseInline(Graph& graph, Gradient& grad_desc,
                                  const std::vector<bool>& input_requires_grad) {
  JIT_ASSERT(graph.stage() == 0);
  graph.advanceStage();

  auto requires_grad_set = findAllRequiresGradNodes(graph, input_requires_grad);
  const auto requires_grad = [&](Value *v) { return requires_grad_set.count(v) > 0; };

  value_map grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) -> Value* {
    auto it = grad_map.find(v);
    if (it == grad_map.end()) {
      std::tie(it, std::ignore) = grad_map.emplace(v, createZerosLike(v));
    }
    return it->second;
  };
  const auto set_grad = [&](Value *x, Value *dx) {
    if (Value * prev_grad = grad_map[x]) {
      Value * new_grad = addValues(prev_grad, dx);
      grad_map[x] = new_grad;
    } else {
      grad_map[x] = dx;
    }
  };

  auto outputs = graph.outputs();
  for (std::size_t i = 0, num_outputs = outputs.size(); i < num_outputs; ++i) {
    Value * output = outputs[i];
    if (!requires_grad(output)) continue;
    Value * output_grad = graph.addInput()->setType(output->typeOption());
    output_grad = createUndefGuard(output_grad, createZerosLike(output));
    set_grad(output, output_grad);
    grad_desc.df_input_vjps.push_back(i);
  }

  for (auto it = graph.nodes().rbegin(), end = graph.nodes().rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    if (std::none_of(inputs.begin(), inputs.end(), requires_grad)) continue;
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
    if (!requires_grad(input)) continue;
    graph.registerOutput(get_grad(input));
    grad_desc.df_output_vjps.push_back(i);
  }

  return ReverseDetails(std::move(grad_map), std::move(requires_grad_set));
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

bool isZero(Value * v) {
  auto n = v->node();
  return n->kind() == kConstant &&
    n->hasAttribute(kis_zero) &&
    n->i(kis_zero);
}

// In the case where an input is routed to an output
// return the (possibly undefined) input rather than
// the value guarded by replaceIfUndef
// this ensures that we do not produce a 0 tensor
// when the autograd would produce None
// graph(a) {
//   b = replaceIfUndef(a,0);
//   c = b + b
//   return c, b; // will replace 'b' with 'a'
// }
// Also replace any known-to-be-zero outputs with Undef
// for the same reason

static void passthroughUndefs(std::shared_ptr<Graph> graph) {
  bool changed = false;
  for(size_t i = 0; i < graph->outputs().size(); i++) {
      Value * v = graph->outputs()[i];
      if(v->node()->kind() == kReplaceIfUndef) {
        graph->return_node()->replaceInput(i, v->node()->inputs()[0]);
        changed = true;
      } else if(isZero(v)) {
        auto undef = graph->appendNode(graph->createUndefined());
        graph->return_node()->replaceInput(i, undef->output());
        changed = true;
      }
  }
  // handle cases where replaceIfUndef or constants has become dead
  if(changed)
    EliminateDeadCode(graph);

}

// Takes a graph returned from `addReverseInline` and splits it into two graphs
// (one for each stage). All intermediates needed in the second stage are added to
// outputs of the first graph, and taken as inputs in the second one. For a more
// detailed description see Note [Gradient graphs] in autodiff.h.
// This function also initializes the fields in grad_desc that were undefined after
// `addReverseInline` (and modifies `df_input_vjps`).
static void lambdaLiftReverse(Graph& graph,
                              ReverseDetails& rev_info,
                              Gradient& grad_desc) {
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
    // Add VJP inputs only for intermediates that actually required grad.
    if (rev_info.requires_grad_set.count(tmp) == 0) continue;
    Value * tmp_vjp_in = graph.addInput()->setType(tmp->typeOption());
    Value * tmp_vjp_prev = rev_info.grad_map.at(tmp);
    auto zeroes = createZerosLike(tmp);
    tmp_vjp_in = createUndefGuard(tmp_vjp_in, zeroes);
    // make sure createUndefGuard happens before the addition but inside stage 1
    zeroes->node()->moveBefore(tmp_vjp_prev->node());
    tmp_vjp_in->node()->moveBefore(tmp_vjp_prev->node());

    // This is quite weird because we can't first make a sum and then replace all uses
    // of tmp_vjp_prev (that would replace its use in the sum too!), so we create an
    // incorrect sum that doesn't use prev vjp, replace uses, and fix the sum.
    Value * new_vjp = addValues(tmp_vjp_in, tmp_vjp_in, tmp_vjp_prev->node());
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

Gradient differentiate(std::shared_ptr<Graph>& _graph, const std::vector<bool>& requires_grad) {
  // Take ownership of the graph
  std::shared_ptr<Graph> graph;
  JIT_ASSERTM(_graph.use_count() == 1,
              "differentiate will mutate and destroy the graph, so it requires "
              "graph.use_count() == 1");
  std::swap(_graph, graph);
  // XXX: Take care when handling outputs - they can be duplicated!
  Gradient grad_desc;
  // Fills in df_input_vjps and df_output_vjps
  auto rev_info = addReverseInline(*graph, grad_desc, requires_grad);
  // addReverseInline has to call gradientForNode if *any* of the outputs
  // require grad, but it will emit vjps for *all* outputs. Use DCE to remove
  // unnecessary nodes.
  EliminateDeadCode(graph);
  // Fills in f, df, f_real_outputs, df_input_captures,
  // modifies df_input_vjps (new vjps are added for temporaries)
  lambdaLiftReverse(*graph, rev_info, grad_desc);
  passthroughUndefs(grad_desc.df);
  return grad_desc;
}

}}
