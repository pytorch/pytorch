#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/tensor.h"

namespace torch { namespace jit { namespace tracer {

namespace detail {

static std::shared_ptr<autograd::Function> insertIdentity(variable_list& vars) {
  int num_vars = vars.size();
  variable_list vars_with_grad;
  for (auto& var : vars) {
    if (var && var->requires_grad)
      vars_with_grad.emplace_back(var);
  }
  auto bw_hook_fn = std::make_shared<autograd::Identity>(autograd::Function::flags(vars_with_grad));
  bw_hook_fn->num_inputs = vars_with_grad.size();
  int output_nr = 0;
  for (int i = 0; i < num_vars; ++i) {
    if (!vars[i]) continue;
    auto& var = vars[i];
    if (!var->requires_grad) continue;
    auto var_clone = var->save(bw_hook_fn.get()).unpack(bw_hook_fn);
    var_clone->grad_fn = bw_hook_fn;
    var_clone->output_nr = output_nr++;
    vars[i] = var_clone;
  }
  return bw_hook_fn;
}

template<typename Subclass>
auto TracerHook<Subclass>::registerHook(
        const std::shared_ptr<TracingState>& tracing_state,
        variable_list& vars) -> std::shared_ptr<Subclass> {
  auto id_fn = insertIdentity(vars);
  // We can't use make_shared, because make_shared is not a friend of Subclass,
  // so it can't use its private constructor...
  auto hook = std::shared_ptr<Subclass>(new Subclass());
  hook->tracing_state = tracing_state;
  id_fn->pre_hooks.emplace_back(hook);
  return hook;
}

////////////////////////////////////////////////////////////////////////////////
// TraceEnterHook
////////////////////////////////////////////////////////////////////////////////

void TraceEnterHook::run(variable_list& vars) {
  auto& graph = tracing_state->graph;
  tracing_state->active = true;
  graph->advanceStage();

  int num_vars = vars.size();
  for (int i = 0; i < num_vars; ++i) {
    auto& var_state = vars[i]->tracing_state;
    auto var_tracing_state = var_state.state.lock();
    if (var_tracing_state) {
      JIT_ASSERT(var_tracing_state == tracing_state);
      // It's ok if an input has a trace from the current stage - we're using a
      // "global" tracing switch, so we might have traced parts we don't care about.
      // We'll replace the trace of this Variable, and DCO will destroy unnecessary nodes.
      // TODO: it might be possible to support values form previous stages being inputs
      // here, although I don't know how right now.
      JIT_ASSERT(var_state.trace->stage() == graph->stage());
    }
    setValueTrace(tracing_state, vars[i], graph->addInput());
  }
  TraceExitHook::registerHook(tracing_state, vars);
}

void TraceEnterHook::registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& outputs) {
  JIT_ASSERT(outputs.size() > 0);

  // Either no (e.g. after last backward) or all outputs should have a grad fn.
  bool has_grad_fn = static_cast<bool>(outputs[0]->grad_fn);
  for (auto& output : outputs) {
    JIT_ASSERT(static_cast<bool>(output->grad_fn) == has_grad_fn);
  }
  if (!has_grad_fn) return;

  auto hook = TracerHook<TraceEnterHook>::registerHook(tracing_state, outputs);
}

////////////////////////////////////////////////////////////////////////////////
// TraceExitHook
////////////////////////////////////////////////////////////////////////////////

void TraceExitHook::run(variable_list& vars) {
  exit(vars);
}

void TraceExitHook::registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& inputs) {
  TracerHook<TraceExitHook>::registerHook(tracing_state, inputs);
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// EvalEnterHook
////////////////////////////////////////////////////////////////////////////////

void EvalEnterHook::run(variable_list& vars) {
  auto& graph = tracing_state->graph;
  Node *eval_node = common_state->eval_node = graph->appendNode(graph->create<Eval>());
  for (auto& input : vars)  {
    eval_node->addInput(tracer::getValueTrace(tracing_state, input, true));
  }
  common_state->next_common_state = EvalExitHook::registerHook(tracing_state, vars);
}

void EvalEnterHook::registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& outputs, std::shared_ptr<EvalCommonState> common_state) {
  auto hook = TracerHook<EvalEnterHook>::registerHook(tracing_state, outputs);
  hook->common_state = common_state;
}

////////////////////////////////////////////////////////////////////////////////
// EvalExitHook
////////////////////////////////////////////////////////////////////////////////

// TODO: handle saved_variable edges. probably need to go through traces of outputs before overwriting
// and find places where they refer to earlier-stage IR
void EvalExitHook::run(variable_list& vars) {
  auto& graph = tracing_state->graph;
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; ++i) {
    auto& var = vars[i];
    auto select = graph->appendNode(graph->create<Select>(common_state->eval_node, i));
    tracer::setValueTrace(tracing_state, var, select);
  }
  EvalEnterHook::registerHook(tracing_state, vars, common_state->next_common_state);
}

std::shared_ptr<EvalCommonState> EvalExitHook::registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& inputs) {
  auto hook = TracerHook<EvalExitHook>::registerHook(tracing_state, inputs);
  hook->common_state = std::make_shared<EvalCommonState>();
  return hook->common_state;
}

}}}
