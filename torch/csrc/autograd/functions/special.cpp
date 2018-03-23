#include "torch/csrc/autograd/functions/special.h"

#include "torch/csrc/assertions.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility> // for swap

namespace torch { namespace autograd {

// Used when an output has multiple uses (there's only one entry
// in next_edges per output).
struct Replicate : public Function {
  Replicate() : Function(/*num_inputs=*/1) {}

  virtual variable_list apply(const variable_list& inputs) {
		TORCH_ASSERT(inputs.size() == 1);
    return variable_list(num_outputs(), inputs[0]);
  }
};

// Note [Null-edge pruning]
// Evals have a problem with null edges appearing in the graph, because there's
// no way to tell the identity of the input (i.e. each nullptr might have been
// a different input, all of them might have been a single input, etc.).
// However, null edges are generally quite useless, so we can safely prune them,
// by removing them from next_edges of Eval node and never allocating
// placeholders for them. This is a bit annoying because backward subgraphs may
// have many less outputs than forward graph had inputs, but I don't think there's
// a way around it. It's a tiny perf optimization too :)

// There's some subtlety involved in computing backwards of Eval functions,
// because sometimes we need to inherit placeholders. There are two situations
// in which it can happen:
// 1. One of the nodes in subgraph saved a Variable, that has a grad_fn that was
//    moved into the interior of the subgraph. Thus, if we were to traverse the
//    graph from an output created when using this Variable, we would end up in
//    one of the placeholders. We don't want this to happen, so we'll inherit it
//    and include the whole subgraph saved grad_fn in this Eval node too (they
//    will be shared, which is ok, because they're immutable at this point).
// 2. One of the nodes in subgraph saved a Variable, that has a grad_fn that
//    points to a node outside of the subgraph (it's grad_fn of one of subgraph's
//    inputs). In this situation, the previous subgraph must have had a placeholder
//    for this input, and we should inherit it as well.
// INVARIANT: all outputs are relevant.
auto Eval::getSubgraph(const variable_list& inputs, const variable_list& outputs,
                       const placeholder_list& inherited_placeholders) -> Subgraph {
  Subgraph subgraph;
  std::unordered_set<std::shared_ptr<EvalOutput>> extra_placeholders;

  // Prepare a set of all edges that shouldn't be followed during the search
  edge_set input_edges;
  input_edges.reserve(inputs.size());
  for (auto & input : inputs) {
    if (!input.defined()) continue;
    input_edges.emplace(input.gradient_edge());
  }

  // This is used to stop the search in situation 2 and find the corresponding placeholders.
  std::unordered_map<Edge, std::shared_ptr<EvalOutput>> inherited_edges;
  inherited_edges.reserve(inherited_placeholders.size());
  for (auto & placeholder : inherited_placeholders) {
    input_edges.emplace(placeholder->next_edge);
    inherited_edges.emplace(placeholder->next_edge, placeholder);
  }

  // Regular DFS data structures
  std::unordered_set<Function*> seen;
  std::vector<Function*> queue;
  for (auto & output : outputs) {
    auto ptr = output.grad_fn().get();
    bool unseen = seen.emplace(ptr).second;
    if (unseen)
      queue.emplace_back(ptr);
  }

  while (!queue.empty()) {
    auto fn = queue.back(); queue.pop_back();
    JIT_ASSERT(fn);
    fn->tracing_state().in_eval_subgraph = true;
    const auto num_outputs = fn->num_outputs();
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& edge = fn->next_edge(i);
      if (!edge.function) continue; // See Note [Null-edge pruning]
      // Edge belongs to subgraph boundary. Register that and don't search along it.
      if (input_edges.count(edge) > 0) {
        subgraph.boundary.begins.emplace(fn->get_shared_ptr(), i);
        subgraph.boundary.ends.emplace(edge);
        auto it = inherited_edges.find(edge);
        // Situation 2. If that edge is actually pointing to an earlier stage subgraph,
        // we'll also need to inherit its placeholder.
        if (it != inherited_edges.end()) {
          extra_placeholders.emplace(it->second);
        }
        continue;
      }
      // Situation 1. If we end up in a placeholder, we need to inherit it.
      if (auto placeholder = std::dynamic_pointer_cast<EvalOutput>(edge.function)) {
        extra_placeholders.emplace(placeholder);
        subgraph.boundary.ends.emplace(placeholder->next_edge);
        continue;
      }
      bool unseen = seen.emplace(edge.function.get()).second;
      if (unseen)
        queue.emplace_back(edge.function.get());
    }
  }

  // Initially fill placeholders with those that we'll need to inherit.
  for (auto & placeholder : extra_placeholders)
    placeholders.emplace_back(placeholder);
  return subgraph;
}

bool Eval::trySimpleEval(const variable_list& inputs, const variable_list& outputs,
                         const placeholder_list& inherited_placeholders) {
  using bitset_type = uint64_t;
  constexpr std::size_t max_outputs = sizeof(bitset_type) * 8;

  if (inherited_placeholders.size() != 0) return false;

  auto& grad_fn = outputs[0].grad_fn();
  if (static_cast<std::size_t>(grad_fn->num_inputs()) >= max_outputs) return false;
  if (static_cast<std::size_t>(grad_fn->num_inputs()) != outputs.size()) return false;

  // Check that all outputs have the same grad_fn and cover all its inputs
  bitset_type output_nrs = 0;
  bitset_type expected_bitset = ((1 << grad_fn->num_inputs()) - 1);
  for (auto & output : outputs) {
    if (output.grad_fn() != grad_fn) return false;
    output_nrs |= (1 << output.output_nr());
  }
  if (output_nrs != expected_bitset) return false;

  // Check that grad_fn's next_edges match the inputs exactly.
  auto num_inputs = inputs.size();
  if (num_inputs != grad_fn->num_outputs()) return false;
  for (std::size_t i = 0; i < num_inputs; ++i) {
    const auto& next_grad_edge = grad_fn->next_edge(i);
    // Unfortunately, null edge pruning (see Note [Null-edge pruning]) applies
    // to autograd functions which would otherwise be eligible for the
    // SimpleEval optimization.  This makes everything more complicated, so for
    // now we just don't attempt the optimization in this case.  To fix it
    // properly, we'd need to filter grad_fn's output edges and outputs of
    // apply in Eval::apply. The check below tests if null edge pruning
    // occurred.
    if (!inputs[i].defined() || !next_grad_edge.is_valid()) return false;
    if (next_grad_edge != inputs[i].gradient_edge()) return false;
  }

  // Success! We still need to set up placeholders for next stages and to drop
  // references to the graph.
  std::swap(next_edges_, grad_fn->next_edges());
  grad_fn->next_edges().reserve(num_inputs);
  placeholders.reserve(num_inputs);
  for (const auto& input : next_edges_) {
    auto placeholder = std::make_shared<EvalOutput>(input);
    grad_fn->add_next_edge({placeholder, 0});
    placeholders.emplace_back(std::move(placeholder));
  }
  simple_graph = grad_fn;
  grad_fn->tracing_state().in_eval_subgraph = true;
  return true;
}


// Here, a _relevant_ output is one that has a grad_fn (is not a leaf and is not
// volatile) and is not one of the inputs (can happen because of passthrough).
variable_list Eval::filterRelevantOutputs(const variable_list& inputs, const variable_list& outputs) {
  variable_list relevant_outputs;
  relevant_outputs.reserve(outputs.size());
  edge_set ignored_grad_fns;
  ignored_grad_fns.reserve(inputs.size());
  for (auto& input : inputs) {
    if (!input.defined()) continue;
    ignored_grad_fns.insert(input.gradient_edge());
  }
  for (auto& output : outputs) {
    if (!output.defined()) continue;
    if (!output.grad_fn()) continue;
    if (ignored_grad_fns.count(output.gradient_edge()) > 0) continue;
    relevant_outputs.emplace_back(output);
  }
  return relevant_outputs;
}

auto Eval::computeInputOrder(const variable_list& inputs, const placeholder_list& inherited_placeholders) -> edge_order {
  edge_order input_order;
  int idx = 0;
  for (auto & input : inputs) {
    if (!input.defined()) continue;
    input_order.emplace(input.gradient_edge(), idx++);
  }
  for (auto & placeholder : inherited_placeholders)
    input_order.emplace(placeholder->next_edge, idx++);
  return input_order;
}

bool Eval::replaceSubgraph(const variable_list& inputs, const variable_list& _outputs,
                           const placeholder_list& inherited_placeholders) {
  // _outputs has a prefix deliberately, because it's unlikely that anything else
  // than relevant_outputs will be needed inside this function.
  // TODO: it would be useful to unpack inputs to their grad_fn/grad_accumulators to avoid
  // all these ternary operators in functions above
  variable_list relevant_outputs = filterRelevantOutputs(inputs, _outputs);

  if (relevant_outputs.size() == 0)
    return false;

  if (!trySimpleEval(inputs, relevant_outputs, inherited_placeholders)) {
    roots.reserve(relevant_outputs.size());
    for (auto & output : relevant_outputs)
      roots.push_back(output.gradient_edge());

    auto subgraph = getSubgraph(inputs, relevant_outputs, inherited_placeholders);

    // Prepare output placeholder nodes for each end.
    std::unordered_map<Edge, std::shared_ptr<EvalOutput>> ends_to_outputs;
    for (auto & placeholder : placeholders) {
      ends_to_outputs[placeholder->next_edge] = placeholder;
    }
    for (auto & end : subgraph.boundary.ends) {
      if (ends_to_outputs.count(end) == 0) {
        placeholders.emplace_back(std::make_shared<EvalOutput>(end));
        ends_to_outputs[end] = placeholders.back();
      }
    }

    // Replace begins with pointers to output nodes.
    // This detaches the subgraph from the full backward graph.
    for (auto& begin : subgraph.boundary.begins) {
      const auto& edge = begin.function->next_edge(begin.input_nr);
      begin.function->set_next_edge(
          begin.input_nr, Edge(ends_to_outputs.at(edge), 0));
    }

    // Replace subgraph with this node.
    next_edges_.insert(next_edges_.begin(), subgraph.boundary.ends.begin(), subgraph.boundary.ends.end());

    // Ensure placeholders and inputs are sorted in the same way.
    edge_order input_order = computeInputOrder(inputs, inherited_placeholders);
    std::sort(next_edges_.begin(), next_edges_.end(), [&input_order](const Edge &a, const Edge &b) {
      return input_order.at(a) < input_order.at(b);
    });
    std::sort(placeholders.begin(), placeholders.end(), [&input_order](const std::shared_ptr<EvalOutput> &a, const std::shared_ptr<EvalOutput> &b) {
      return input_order.at(a->next_edge) < input_order.at(b->next_edge);
    });
  }

  // Rebase outputs.
  auto this_shared = shared_from_this();
  std::unordered_set<Variable*> repeated_outputs;
  // NB: every output can be in 3 states:
  // - unique so far - only the else of second if is taken
  // - repeated first time - first if + first branch of second if
  // - repeated many times - first branch of second if only
  for (auto & output : relevant_outputs) {
    // This output is already rebased. This happens when there
    // the same Variable has been returned multiple times, and
    // is repeated in this list.
    if (output.grad_fn_unsafe() == this) {
      auto replicate = std::make_shared<Replicate>();
      replicate->add_next_edge({this_shared, output.output_nr()});
      output.set_gradient_edge({std::move(replicate), 0});
      repeated_outputs.emplace(&output);
    }
    // NOTE: this check should be fairly cheap, and the set shouldn't
    // perform any allocations until we actually see repeated outputs.
    if (repeated_outputs.count(&output) > 0) {
      auto & replicate = output.grad_fn();
      replicate->add_next_edge({this_shared, num_inputs_++});
    } else {
      autograd::create_gradient_edge(output, this_shared);
    }
  }

  return true;
}

variable_list Eval::apply(const variable_list& inputs) {
  variable_list outputs;
  if (simple_graph) {
    outputs = (*simple_graph)(inputs);
  } else {
    auto& engine = Engine::getDefaultEngine();
    auto exec_data = filterRoots(inputs);
    auto next_edges = fmap(
        placeholders,
        [](const std::shared_ptr<EvalOutput>& o) { return Edge(o, 0); });
    outputs = engine.execute(exec_data.first, exec_data.second, true, true, next_edges);
  }

  auto bw_eval = newEval();
  bw_eval->replaceSubgraph(inputs, outputs, placeholders);

  // This will prevent Function::traced_apply from marking the backward subgraph as non-traceable.
  // This node already does it (backward of non-traceable backward is implicitly non-traceable),
  // and it passes more information (backward Eval may inherit placeholders) than
  // Function::traced_apply has available.
  tracing_state_->in_eval_subgraph = true;

  return outputs;
}

// TODO: once we clean up the stochastic function mess it should be possible to ignore
// nullptr inputs in the Engine (it implies that the Variables is 0, so the jacobian vector
// product will be all zero too).
std::pair<edge_list, variable_list> Eval::filterRoots(const variable_list& inputs) {
  variable_list filtered_inputs;
  edge_list filtered_roots;
  auto num_inputs = inputs.size();
  if (roots.size() != num_inputs)
    throw std::logic_error("inputs.size() != roots.size()");
  filtered_inputs.reserve(num_inputs);
  filtered_roots.reserve(num_inputs);
  for (std::size_t i = 0; i < num_inputs; ++i) {
    // This check is the sole reason why this function is needed. The problem
    // with larger Evals is that they might trigger computation of nodes that
    // would normally be ignored. For example, consider a subgraph with multiple
    // outputs and a backprop from a Variable that's derived from only one of
    // them. This line prevents us from unnecessarily executing, and thus recording,
    // nodes in the trace which are unrelated to this Variable.
    //
    // If we didn't filter out roots that only get nullptr outputs, we would then
    // pass nullptr inputs to roots that are executable. Then, the engine would
    // discover them and would unnecessarily run a computation that doesn't contribute
    // to the overall grad and would complicate the trace.
    // If the node gets only nullptr inputs, it's guaranteed that
    // the grad of its output w.r.t. anything is 0, so it is sound to just
    // skip the computation entirely.
    if (!inputs[i].defined()) continue;
    filtered_inputs.emplace_back(inputs[i]);
    filtered_roots.emplace_back(roots[i]);
  }
  return std::make_pair(std::move(filtered_roots), std::move(filtered_inputs));
}

}} // namespace torch::autograd
