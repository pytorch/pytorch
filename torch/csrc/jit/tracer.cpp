#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/variable_tensor_functions.h"

#include <string>
#include <sstream>
#include <memory>

namespace torch { namespace jit { namespace tracer {

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
