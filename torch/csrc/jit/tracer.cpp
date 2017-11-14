#include "Python.h"
#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <frameobject.h>
#include <patchlevel.h>

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
      Value *input_node = graph->addInput();
      if (!input.defined()) continue;
      JIT_ASSERT(!detail::getValueState(tracing_state, input, false));
      setValueTrace(tracing_state, input, input_node);
      input_node->inferTypeFrom(input.data());
    }
    tracing_state->var_flags.at(graph->stage()).first = detail::getVarFlags(inputs);
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

namespace {
// Python interpreter retrieval routine adapted from
// https://stackoverflow.com/a/8706144
std::string getPythonInterpreterStackTrace() {
  std::stringstream stack_trace;
  AutoGIL gil;
  PyThreadState *tstate = PyThreadState_GET();
  if (NULL != tstate && NULL != tstate->frame) {
    PyFrameObject *frame = tstate->frame;

    while (NULL != frame) {
      int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      std::string filename = THPUtils_unpackString(frame->f_code->co_filename);
      std::string funcname = THPUtils_unpackString(frame->f_code->co_name);
      stack_trace << filename << "(" << line << "): " << funcname << "\n";
      frame = frame->f_back;
    }
  }
  return stack_trace.str();
}
}  // namespace

Node* recordTrace(std::string op, // TODO: make this a Symbol
                  at::ArrayRef<Variable> inputs,
                  at::ArrayRef<Variable> outputs) {
  auto state = getTracingState(inputs);
  auto& graph = state->graph;
  // TODO: Technically, we could reduce the scope of the lock, but since we
  // haven't actually specified what the locking contract is, be conservative.
  auto state_lock = state->lock();

  Node *n = graph->create(stringToSymbol(op), 0 /* initial outputs */);
  auto sl = std::make_shared<SourceLocation>(getPythonInterpreterStackTrace());
  n->setSourceLocation(sl);

  for (Variable input : inputs) {
    n->addInput(getValueTrace(state, input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  auto assignOutput = [&state](const Variable & output, Value * value) {
    if (output.defined()) {
      value->inferTypeFrom(output.data());
      setValueTrace(state, output, value);
    }
  };

  for(size_t i = 0; i < outputs.size(); i++) {
    assignOutput(outputs[i], n->addOutput());
  }

  // Return the n so that attributes can be added.
  return n;
}

}}}
