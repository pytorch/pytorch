#include "torch/csrc/autograd/functions/special.h"

#include "torch/csrc/autograd/python_engine.h"

namespace torch { namespace autograd {

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
    // TODO: Remove me when we get rid of null Variables
    // See https://github.com/pytorch/pytorch/issues/2513
    if (!input.defined()) continue;
    input_edges.emplace(input.grad_fn() ? input.grad_fn() : input.grad_accumulator(), input.output_nr());
  }

  // This is used to stop the search in situation 2 and find the corresponding placeholders.
  std::unordered_map<edge_type, std::shared_ptr<EvalOutput>, edge_hasher> inherited_edges;
  inherited_edges.reserve(inherited_placeholders.size());
  for (auto & placeholder : inherited_placeholders) {
    input_edges.emplace(placeholder->next_edge);
    inherited_edges.emplace(placeholder->next_edge, placeholder);
  }

  // Regular DFS data structures
  std::unordered_set<Function*> seen { nullptr };
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
    fn->tracing_state->in_eval_subgraph = true;
    int num_edges = fn->next_functions.size();
    for (int i = 0; i < num_edges; ++i) {
      auto & edge = fn->next_functions[i];
      auto & next_fn = edge.first;
      // Edge belongs to subgraph boundary. Register that and don't search along it.
      if (input_edges.count(edge) > 0) {
        subgraph.boundary.begins.emplace(fn->getSharedPtr(), i);
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
      if (auto placeholder = std::dynamic_pointer_cast<EvalOutput>(next_fn)) {
        extra_placeholders.emplace(placeholder);
        subgraph.boundary.ends.emplace(placeholder->next_edge);
        continue;
      }
      bool unseen = seen.emplace(next_fn.get()).second;
      if (unseen)
        queue.emplace_back(next_fn.get());
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

  auto& grad_fn = outputs[0]->grad_fn;
  if (static_cast<std::size_t>(grad_fn->num_inputs) >= max_outputs) return false;
  if (static_cast<std::size_t>(grad_fn->num_inputs) != outputs.size()) return false;

  // Check that all outputs have the same grad_fn and cover all its inputs
  bitset_type output_nrs = 0;
  bitset_type expected_bitset = ((1 << grad_fn->num_inputs) - 1);
  for (auto & output : outputs) {
    if (output->grad_fn != grad_fn) return false;
    output_nrs |= (1 << output->output_nr);
  }
  if (output_nrs != expected_bitset) return false;

  // Check that grad_fn->next_functions matches the inputs exactly
  auto num_inputs = inputs.size();
  auto& grad_next_fns = grad_fn->next_functions;
  if (num_inputs != grad_next_fns.size()) return false;
  for (std::size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i]) {
      const auto& input_grad = inputs[i]->grad_fn ? inputs[i]->grad_fn : inputs[i]->grad_accumulator.lock();
      if (grad_next_fns[i].first != input_grad || grad_next_fns[i].second != inputs[i]->output_nr) return false;
    } else {
      if (grad_next_fns[i].first != nullptr) return false;
    }
  }

  // Success! We still need to set up placeholders for next stages and to drop
  // references to the graph.
  next_functions = std::move(grad_next_fns);
  grad_next_fns.reserve(num_inputs);
  placeholders.reserve(num_inputs);
  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto placeholder = std::make_shared<EvalOutput>(next_functions[i]);
    grad_next_fns.emplace_back(placeholder, 0);
    placeholders.emplace_back(std::move(placeholder));
  }
  is_executable = grad_fn->is_executable;
  simple_graph = grad_fn;
  input_sizes = grad_fn->input_sizes;
  grad_fn->tracing_state->in_eval_subgraph = true;
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
    // TODO: Remove me when we get rid of null Variables
    // See https://github.com/pytorch/pytorch/issues/2513
    if (!input.defined()) continue;
    ignored_grad_fns.emplace(input.grad_fn(), input.output_nr());
  }
  for (auto& output : outputs) {
    // TODO: Remove me when we get rid of null Variables
    // See https://github.com/pytorch/pytorch/issues/2513
    if (!output.defined()) continue;
    if (!output.grad_fn()) continue;
    if (ignored_grad_fns.count(std::make_pair(output.grad_fn(), output.output_nr())) > 0) continue;
    relevant_outputs.emplace_back(output);
  }
  return relevant_outputs;
}

auto Eval::computeInputOrder(const variable_list& inputs, const placeholder_list& inherited_placeholders) -> edge_order {
  edge_order input_order;
  int idx = 0;
  for (auto & input : inputs) {
    // TODO: Remove me when we get rid of null Variables
    // See https://github.com/pytorch/pytorch/issues/2513
    if (!input.defined()) continue;
    input_order.emplace(
      std::make_pair(input.grad_fn() ? input.grad_fn() : input.grad_accumulator(), input.output_nr()),
      idx++
    );
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
      roots.emplace_back(output.grad_fn(), output.output_nr());

    auto subgraph = getSubgraph(inputs, relevant_outputs, inherited_placeholders);

    // Prepare output placeholder nodes for each end.
    std::unordered_map<edge_type, std::shared_ptr<EvalOutput>, edge_hasher> ends_to_outputs;
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
    for (auto & begin : subgraph.boundary.begins) {
      auto & fn = begin.first;
      auto offset = begin.second;
      fn->next_functions[offset] = std::make_pair(ends_to_outputs.at(fn->next_functions[offset]), 0);
    }

    // Replace subgraph with this node.
    next_functions.insert(next_functions.begin(), subgraph.boundary.ends.begin(), subgraph.boundary.ends.end());
    is_executable = std::any_of(relevant_outputs.begin(), relevant_outputs.end(),
                                [](const Variable& var) { return var.requires_grad(); });
    input_sizes = fmap<TensorMeta>(relevant_outputs);

    // Ensure placeholders and inputs are sorted in the same way.
    edge_order input_order = computeInputOrder(inputs, inherited_placeholders);
    std::sort(next_functions.begin(), next_functions.end(), [&input_order](const edge_type &a, const edge_type &b) {
      return input_order.at(a) < input_order.at(b);
    });
    std::sort(placeholders.begin(), placeholders.end(), [&input_order](const std::shared_ptr<EvalOutput> &a, const std::shared_ptr<EvalOutput> &b) {
      return input_order.at(a->next_edge) < input_order.at(b->next_edge);
    });
  }

  // Rebase outputs.
  for (auto & output : relevant_outputs) {
    output.grad_fn() = shared_from_this();
    output.get()->output_nr = num_inputs++;
  }

  return true;
}

variable_list Eval::apply(const variable_list& inputs) {
  variable_list outputs;
  if (simple_graph) {
    outputs = (*simple_graph)(inputs);
  } else {
    std::mutex outputs_mutex;
    outputs.resize(placeholders.size());
    auto& engine = python::PythonEngine::getDefaultEngine();
    auto exec_data = filterRoots(inputs);
    engine.execute(exec_data.first, exec_data.second, true, getCallbacks(outputs, outputs_mutex));
  }

  auto bw_eval = newEval();
  bw_eval->replaceSubgraph(inputs, outputs, placeholders);

  // This will prevent Function::traced_apply from marking the backward subgraph as non-traceable.
  // This node already does it (backward of non-traceable backward is implicitly non-traceable),
  // and it passes more information (backward Eval may inherit placeholders) than
  // Function::traced_apply has available.
  tracing_state->in_eval_subgraph = true;

  return outputs;
}

// TODO: once we clean up the stochastic function mess it should be possible to ignore
// nullptr inputs in the Engine (it implies that the Variables is 0, so the jacobian vector
// product will be all zero too).
std::pair<function_list, variable_list> Eval::filterRoots(const variable_list& inputs) {
  variable_list filtered_inputs;
  function_list filtered_roots;
  auto num_inputs = inputs.size();
  if (roots.size() != num_inputs)
    throw std::logic_error("inputs.size() != roots.size()");
  filtered_inputs.reserve(num_inputs);
  filtered_roots.reserve(num_inputs);
  for (std::size_t i = 0; i < num_inputs; ++i) {
    if (!inputs[i].defined()) continue;
    filtered_inputs.emplace_back(inputs[i]);
    filtered_roots.emplace_back(roots[i]);
  }
  return std::make_pair(std::move(filtered_roots), std::move(filtered_inputs));
}

Engine::pre_callback_map Eval::getCallbacks(variable_list& outputs, std::mutex& outputs_mutex) {
  Engine::pre_callback_map callbacks;
  int num_outputs = placeholders.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output_fn = placeholders[i];
    callbacks.emplace(output_fn.get(), [&outputs, &outputs_mutex, i](Function* _unused, variable_list& inputs) -> bool {
      if (inputs.size() != 1)
        throw std::logic_error("placeholder callback received too many inputs");
      std::lock_guard<std::mutex> lock(outputs_mutex);
      outputs[i] = inputs[0];
      return false; // Stop at output nodes
    });
  }
  return callbacks;
}

}} // namespace torch::autograd
