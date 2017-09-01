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

  std::weak_ptr<TracingState> weak_tracing_state;
};

struct TraceEnterHook : autograd::FunctionPreHook {
  TraceEnterHook(const std::shared_ptr<TracingState>& tracing_state)
    : weak_tracing_state(tracing_state) {}

  virtual variable_list operator()(const variable_list& inputs) {
    // TODO: this shouldn't be run only once - it might be possible that this stage is
    // called multiple times, but e.g. only the second call is unrolled for as many
    // stages as we want.
    std::call_once(flag, &TraceEnterHook::enterTrace, this, std::ref(inputs));
    return inputs;
  }

  void enterTrace(const variable_list& inputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    auto& graph = tracing_state->graph;
    tracing_state->active = true;
    graph->advanceStage();

    for (auto & input : inputs) {
      JIT_ASSERT(!detail::getValueState(tracing_state, input, false));
      Node *input_node = graph->addInput();
      setValueTrace(tracing_state, input, input_node);
      input_node->inferTypeFrom(input->data);
    }
    tracing_state->var_flags[graph->stage()] = detail::getVarFlags(inputs);
  }

  std::weak_ptr<TracingState> weak_tracing_state;
  std::once_flag flag;
};

struct TraceExitHook : autograd::FunctionPostHook {
  TraceExitHook(const std::shared_ptr<TracingState>& tracing_state)
    : weak_tracing_state(tracing_state) {}

  virtual variable_list operator()(const variable_list& outputs, const variable_list& inputs) {
    std::call_once(flag, &TraceExitHook::exitTrace, this, std::ref(inputs), std::ref(outputs));
    return outputs;
  }

  void exitTrace(const variable_list& inputs, const variable_list& outputs) {
    auto tracing_state = weak_tracing_state.lock();
    if (!tracing_state) return;

    detail::_exit(tracing_state, outputs);
    // Unfortunately there's no easy way to get handle of the backward node for current Eval.
    auto eval_fn = autograd::Eval::getBackwardEval(inputs, outputs);
    if (!eval_fn) return;
    eval_fn->pre_hooks.emplace_back(std::make_shared<TraceEnterHook>(tracing_state));
    eval_fn->post_hooks.emplace_back(std::make_shared<TraceExitHook>(tracing_state));
  }

  std::weak_ptr<TracingState> weak_tracing_state;
  std::once_flag flag;
};

} // anonymous namespace

namespace detail {

void traceBackward(const std::shared_ptr<TracingState>& tracing_state, const variable_list& inputs, const variable_list& outputs) {
  // TODO: add note on how we depend on TracedEval being created in here
  auto eval_fn = std::make_shared<TraceEval>(tracing_state);
  eval_fn->replaceSubgraph(inputs, outputs);
  eval_fn->pre_hooks.emplace_back(std::make_shared<TraceEnterHook>(tracing_state));
  eval_fn->post_hooks.emplace_back(std::make_shared<TraceExitHook>(tracing_state));
}

} // namespace detail

void nontraceableBackwardSubgraph(const variable_list& inputs, const variable_list& outputs) {
  std::make_shared<autograd::Eval>()->replaceSubgraph(inputs, outputs);
}

}}}
