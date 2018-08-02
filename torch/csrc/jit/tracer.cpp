#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/jit/assertions.h"
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

////////////////////////////////////////////////////////////////////////////////
// Recording the traces
////////////////////////////////////////////////////////////////////////////////
namespace detail {

template<typename T>
void genericAddInput(Node *n, T value) {
  n->addInput(insertConstant(*n->owningGraph(), value));
}

void badArgType() {
  throw std::runtime_error("Found an unsupported argument type in the JIT tracer. File a bug report.");
}


void addInputs(Node *n, const char * name, int64_t value)            { genericAddInput(n, value); }
void addInputs(Node *n, const char * name, bool value)               { genericAddInput(n, value); }
void addInputs(Node *n, const char * name, double value)             { genericAddInput(n, value); }
void addInputs(Node *n, const char * name, const at::Scalar& value)  { genericAddInput(n, value); }
void addInputs(Node *n, const char * name, const at::Tensor& value)  { n->addInput(getValueTrace(value)); }
void addInputs(Node *n, const char * name, const std::string& value)         { badArgType(); }
void addInputs(Node *n, const char * name, const at::SparseTensorRef& value) { badArgType(); }

void addInputs(Node *n, const char * name, at::TensorList value) {
  Graph *g = n->owningGraph();
  Node *list_node = g->appendNode(g->createList(DynamicType::get(), fmap(value, getValueTrace)));
  n->addInput(list_node->output());
}

void addInputs(Node *n, const char * name, at::IntList value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  std::vector<Value*> info = ArgumentStash::hasIntList(name) ?
    ArgumentStash::popIntList(name) :
    ArgumentStash::IntListTrace(value.size());

  auto& g = getTracingState()->graph;
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i] != nullptr) continue;
    info[i] = insertConstant(*g, value[i]);
  }
  for (jit::Value* v : info) {
    if (*v->type() != *jit::IntType::get()) {
      throw std::runtime_error(
        "Type mismatch in setposattr for IntList. Check that your program "
        "is valid without tracing, and please file a bug report if it is.");
    }
  }
  n->addInput(g->insertNode(g->createList(jit::IntType::get(), info))->output());
}

thread_local std::shared_ptr<TracingState> tracing_state;

} // namespace detail

const std::shared_ptr<TracingState>& getTracingState() {
  return detail::tracing_state;
}

void setTracingState(std::shared_ptr<TracingState> state) {
  detail::tracing_state = std::move(state);
}

TracingState::TracingState()
    : graph(new Graph()) {}

TracingState::~TracingState() = default;

void postRecordTrace(const PreTraceInfo& info,
                     at::ArrayRef<Variable> outputs) {
  for (size_t i = 0; i < outputs.size(); i++) {
    auto & output = outputs[i];
    Value * value = info.n->addOutput();
    if (output.defined()) {
      value->inferTypeFrom(output.data());
      setValueTrace(output, value);
    }
  }
}

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  auto & tracing_state = getTracingState();
  auto & graph = tracing_state->graph;

  auto size_var = autograd::make_variable(at::Scalar(var.size(dim)).toTensor());
  auto* value = getValueTrace(var);
  WithInsertPoint ipoint { graph->block() };
  auto* node = graph->insertNode(graph->create(aten::size, {value, insertConstant(*graph, dim)}));
  node->output()->setType(jit::IntType::get());

  auto ten =
      graph->appendNode(graph->createNumToTensor(node->output()))->output();
  setValueTrace(size_var, ten);
  return size_var;
}

////////////////////////////////////////////////////////////////////////////////
// Argument stash
////////////////////////////////////////////////////////////////////////////////
thread_local ArgumentStash ArgumentStash::stash;

void ArgumentStash::stashIntListElem(const std::string& arg_name, size_t size, size_t idx, const Variable& var) {
  // TODO: check type?
  if (!isTracing()) return;
  auto & list_trace = stash.intlists.emplace(arg_name, size).first->second;
  JIT_ASSERT(size == list_trace.size());
  JIT_ASSERT(idx < list_trace.size());
  JIT_ASSERT(list_trace[idx] == nullptr);

  Value* ten = getValueTrace(var);
  auto& g = *ten->owningGraph();
  auto prim = g.createTensorToNum(jit::IntType::get(), ten)
                   ->insertAfter(ten->node())
                   ->output();
  list_trace[idx] = prim;
}

////////////////////////////////////////////////////////////////////////////////
// Stack trace recording
////////////////////////////////////////////////////////////////////////////////
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
