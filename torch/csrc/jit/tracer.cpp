#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/functions/special.h"

namespace torch { namespace jit { namespace tracer {

namespace {

struct TraceEval : autograd::Eval {
  TraceEval(const std::shared_ptr<TracingState>& tracing_state)
    : weak_tracing_state(tracing_state) {
    flag.clear();
    tracing_state->eval_count++;
    this->traceable = true;
  }

  virtual ~TraceEval() {
    auto state = weak_tracing_state.lock();
    if (!state) return;
    if (--state->eval_count == 0 && !state->is_complete()) {
      state->graph = nullptr;
    }
  }

  virtual std::shared_ptr<Eval> newEval() override {
    if (auto state = weak_tracing_state.lock()) {
      return std::make_shared<TraceEval>(state);
    } else {
      return std::make_shared<autograd::Eval>();
    }
  }

  virtual variable_list apply(const variable_list& inputs) override {
    auto should_trace = !flag.test_and_set();
    if (should_trace) enterTrace(inputs);
    auto outputs = Eval::apply(inputs);
    if (should_trace) exitTrace(inputs, outputs);
    return outputs;
  }

  void enterTrace(const variable_list& inputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    auto& graph = tracing_state->graph;
    tracing_state->active = true;
    graph->advanceStage();

    for (auto & input : inputs) {
      Node *input_node = graph->addInput();
      if (!input.defined()) continue;
      JIT_ASSERT(!detail::getValueState(tracing_state, input, false));
      setValueTrace(tracing_state, input, input_node);
      input_node->inferTypeFrom(input.data());
    }
    tracing_state->var_flags.at(graph->stage()) = detail::getVarFlags(inputs);
  }

  void exitTrace(const variable_list& inputs, const variable_list& outputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    detail::_exit(tracing_state, outputs);
    auto stage = tracing_state->graph->stage();
    tracing_state->output_edges[stage] = fmap(placeholders, [](const std::shared_ptr<autograd::EvalOutput> p) {
      return p->next_edge;
    });
  }

  std::atomic_flag flag;
  std::weak_ptr<TracingState> weak_tracing_state;
};

} // anonymous namespace

namespace detail {

void traceBackward(const std::shared_ptr<TracingState>& tracing_state, const variable_list& inputs, const variable_list& outputs) {
  // TODO: add note on how we depend on TracedEval being created in here if num_stages == 1
  std::make_shared<TraceEval>(tracing_state)->replaceSubgraph(inputs, outputs);
}

} // namespace detail

void nontraceableBackwardSubgraph(const variable_list& inputs, const variable_list& outputs) {
  std::make_shared<autograd::Eval>()->replaceSubgraph(inputs, outputs);
}

Node* recordTraceHelper(std::string op, // TODO: make this a Symbol
                        at::ArrayRef<Variable> inputs,
                        at::ArrayRef<Variable> outputs) {
  auto state = getTracingState(inputs);
  auto& graph = state->graph;
  // TODO: Technically, we could reduce the scope of the lock, but since we
  // haven't actually specified what the locking contract is, be conservative.
  auto state_lock = state->lock();

  Node *n = graph->create(stringToSymbol(op));
  for (Variable input : inputs) {
    n->addInput(getValueTrace(state, input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  int i = 0;
  for (Variable output : outputs) {
    Node* sel = graph->appendNode(graph->createSelect(n, i));
    // TODO: Track inplace operations (needed for JIT).
    if (output.defined()) {
      sel->inferTypeFrom(output.data());
      setValueTrace(state, output, sel);
    }
    i++;
  }

  /*
  // Want to do this eventually, but we also need to deal with Handles in this
  // world
  if (outputs.size() != 1) {
    int i = 0;
    for (Variable output : outputs) {
      Node* sel = graph->appendNode(graph->createSelect(n, i));
      // TODO: Track inplace operations (needed for JIT).
      if (output.defined()) {
        sel->inferTypeFrom(output.data());
        setValueTrace(state, output, sel);
      }
      i++;
    }
  } else {
    setValueTrace(state, outputs.front(), n);
  }
  */

  // Return the n so that attributes can be added.
  return n;
}

}}}
