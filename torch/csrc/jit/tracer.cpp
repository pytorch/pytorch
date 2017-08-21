#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/functions/special.h"

namespace torch { namespace jit { namespace tracer {

void nontraceableBackwardSubgraph(const variable_list& inputs, const variable_list& outputs) {
  std::make_shared<autograd::Eval>()->replaceSubgraph(inputs, outputs);
}

namespace detail {

struct TraceEnterHook : autograd::FunctionPreHook {
  TraceEnterHook(const std::shared_ptr<TracingState>& tracing_state)
    : tracing_state(tracing_state) {}

  virtual variable_list operator()(const variable_list& inputs) {
    std::call_once(flag, &TraceEnterHook::enterTrace, this, std::ref(inputs));
    return inputs;
  }

  void enterTrace(const variable_list& inputs) {
    auto& graph = tracing_state->graph;
    tracing_state->active = true;
    graph->advanceStage();

    for (auto & input : inputs) {
      JIT_ASSERT(input->tracing_state.state.expired());
      Node *input_node = graph->addInput();
      setValueTrace(tracing_state, input, input_node);
      input_node->inferTypeFrom(input->data);
    }
  }

  std::shared_ptr<TracingState> tracing_state;
  std::once_flag flag;
};

struct TraceExitHook : autograd::FunctionPostHook {
  TraceExitHook(const std::shared_ptr<TracingState>& tracing_state)
    : tracing_state(tracing_state) {}

  virtual variable_list operator()(const variable_list& outputs, const variable_list& inputs) {
    std::call_once(flag, &TraceExitHook::exitTrace, this, std::ref(inputs), std::ref(outputs));
    return outputs;
  }

  void exitTrace(const variable_list& inputs, const variable_list& outputs) {
    detail::_exit(tracing_state, outputs);
    // Unfortunately there's no easy way to get handle of the backward node for current Eval.
    auto eval_fn = autograd::Eval::getBackwardEval(inputs, outputs);
    if (!eval_fn) return;
    eval_fn->pre_hooks.emplace_back(std::make_shared<TraceEnterHook>(tracing_state));
    eval_fn->post_hooks.emplace_back(std::make_shared<TraceExitHook>(tracing_state));
    eval_fn->traceable = true;
  }

  std::shared_ptr<TracingState> tracing_state;
  std::once_flag flag;
};

void traceBackward(const std::shared_ptr<TracingState>& tracing_state, const variable_list& inputs, const variable_list& outputs) {
  auto eval_fn = std::make_shared<autograd::Eval>();
  eval_fn->replaceSubgraph(inputs, outputs);
  eval_fn->traceable = true;
  eval_fn->pre_hooks.emplace_back(std::make_shared<TraceEnterHook>(tracing_state));
  eval_fn->post_hooks.emplace_back(std::make_shared<TraceExitHook>(tracing_state));
}

} // namespace detail

}}}
