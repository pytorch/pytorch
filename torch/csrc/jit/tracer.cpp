#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/variable_tensor_functions.h"

#include <string>
#include <sstream>
#include <memory>

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

    for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
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

PreTraceInfo preRecordTrace(Symbol op,
                            at::ArrayRef<Variable> inputs) {
  return makePreTraceInfo(inputs, [&op](const std::shared_ptr<TracingState>& state, Graph& graph) {
    return graph.create(op, 0 /* initial outputs */);
  });
}

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

thread_local ArgumentStash ArgumentStash::stash;

void ArgumentStash::stashIntListElem(const std::string& arg_name, size_t size, size_t idx, const Variable& var) {
  // TODO: check type?
  if (!isTracing(var)) return;
  auto tracing_state = getTracingState({var});
  auto & list_trace = stash.intlists.emplace(arg_name, size).first->second;
  JIT_ASSERT(size == list_trace.size());
  JIT_ASSERT(idx < list_trace.size());
  JIT_ASSERT(list_trace[idx] == nullptr);
  list_trace[idx] = getValueTrace(tracing_state, var);
}

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  auto tracing_state = getTracingState({var});
  auto & graph = tracing_state->graph;

  auto size_var = autograd::make_variable(at::Scalar(var.size(dim)).toTensor());
  auto* value = getValueTrace(tracing_state, var);
  auto* node = graph->create(aten::size, {value})
                    ->i_(attr::dim, dim);
  node->output()->inferTypeFrom(size_var);
  graph->appendNode(node);
  setValueTrace(tracing_state, size_var, node->output());

  return size_var;
}


// no python present so we just do not record source information
void defaultRecordSourceLocation(Node* n) {}
std::atomic<decltype(&defaultRecordSourceLocation)> record_source_location(defaultRecordSourceLocation);
void recordSourceLocation(Node* n) {
  return record_source_location.load()(n);
}
void setRecordSourceLocation(void (*v)(Node*)) {
  record_source_location.store(v);
}

}}}
