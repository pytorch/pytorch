#ifndef NO_PYTHON
#include <Python.h>
#endif
#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/functions/special.h"

#include <string>
#include <sstream>
#include <memory>

#ifndef NO_PYTHON
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include <frameobject.h>
#include <patchlevel.h>

// Python interpreter retrieval routine adapted from
// https://stackoverflow.com/a/8706144
std::string torch::jit::tracer::getPythonInterpreterStackTrace() {
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
#endif

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
    if (!should_trace) {
      return Eval::apply(inputs);
    }
    variable_list local_inputs = inputs;
    enterTrace(local_inputs);
    auto outputs = Eval::apply(local_inputs);
    exitTrace(local_inputs, outputs);
    return outputs;
  }

  void enterTrace(variable_list& inputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    auto& graph = tracing_state->graph;
    graph->advanceStage();

    for (std::size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
      auto input = inputs[i];
      Value *input_node = graph->addInput();
      if (!input.defined()) continue;
      auto * value_state = detail::getValueState(tracing_state, input, false);
      if (value_state) {
        // Note [Repeated inputs]
        // Repeated inputs cause us some problems in here, because there's no way
        // for us to attach a single Variable to two inputs, and to tell which one
        // is used when performing an operation. To deal with it, we allocate a view
        // of such input, and use that instead.
        inputs[i] = input = input.view(input.sizes());
      }
      setValueTrace(tracing_state, input, input_node);
      input_node->inferTypeFrom(input.data());
    }
    tracing_state->active = true;
    tracing_state->var_flags.at(graph->stage()).first = detail::getVarFlags(inputs);
  }

  void exitTrace(const variable_list& inputs, const variable_list& outputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    detail::_exit(tracing_state, outputs);
    auto stage = tracing_state->graph->stage();
    tracing_state->output_edges[stage] = fmap(placeholders, [](const std::shared_ptr<autograd::EvalOutput>& p) {
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

// We must record the nodes of inputs before we actually carry out
// the operation, because an inplace operation may destroy the information
// we're interested in.  See #4480.
template<typename F>
PreTraceInfo makePreTraceInfo(at::ArrayRef<Variable> inputs, F ctor) {
  PreTraceInfo info;
  info.state = getTracingState(inputs);
  auto& graph = info.state->graph;
  auto state_lock = info.state->lock();

  Node *n = ctor(*graph);
#ifndef NO_PYTHON
  auto sl = std::make_shared<StringSourceLocation>(getPythonInterpreterStackTrace());
  n->setSourceLocation(sl);
#endif

  for (Variable input : inputs) {
    n->addInput(getValueTrace(info.state, input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  info.n = n;

  return info;
}

PreTraceInfo preRecordTrace(std::string op, // TODO: make this a Symbol
                            at::ArrayRef<Variable> inputs) {
  return makePreTraceInfo(inputs, [&op](Graph& graph) {
    return graph.create(Symbol(op), 0 /* initial outputs */);
  });
}

#ifndef NO_PYTHON
PreTraceInfo preRecordPythonTrace(THPObjectPtr pyobj,
                                  std::string arg_types,
                                  at::ArrayRef<Variable> inputs,
                                  pyobj_list scalar_args) {
  std::vector<VariableFlags> var_flags = fmap(inputs, &VariableFlags::of);

  return makePreTraceInfo(inputs, [&](Graph& graph) {
    const bool is_legacy = false;
    return graph.createPythonOp(
        std::move(pyobj),
        arg_types,
        is_legacy,
        std::move(var_flags),
        std::move(scalar_args));
  });
}
#endif

void postRecordTrace(const PreTraceInfo& info,
                     at::ArrayRef<Variable> outputs) {
  // TODO: Technically, we could reduce the scope of the lock, but since we
  // haven't actually specified what the locking contract is, be conservative.
  auto state_lock = info.state->lock();

  auto assignOutput = [&info](const Variable & output, Value * value) {
    if (output.defined()) {
      value->inferTypeFrom(output.data());
      setValueTrace(info.state, output, value);
    }
  };

  for (size_t i = 0; i < outputs.size(); i++) {
    assignOutput(outputs[i], info.n->addOutput());
  }
}

}}}
