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
  Value *v = n->owningGraph()->insertConstant(value);
  recordSourceLocation(v->node());
  n->addInput(v);
}

void badArgType() {
  AT_ERROR("Found an unsupported argument type in the JIT tracer. File a bug report.");
}

thread_local std::shared_ptr<TracingState> tracing_state;

} // namespace detail

void addInputs(Node *n, const char * name, int64_t value)            { detail::genericAddInput(n, value); }
void addInputs(Node *n, const char * name, bool value)               { detail::genericAddInput(n, value); }
void addInputs(Node *n, const char * name, double value)             { detail::genericAddInput(n, value); }
void addInputs(Node *n, const char * name, const at::Scalar& value)  { detail::genericAddInput(n, value); }
void addInputs(Node *n, const char * name, const at::Tensor& value)  { n->addInput(getValueTrace(value)); }
void addInputs(Node *n, const char * name, const std::string& value)         { detail::badArgType(); }
void addInputs(Node *n, const char * name, const at::SparseTensorRef& value) { detail::badArgType(); }

void addInputs(Node *n, const char * name, at::TensorList value) {
  Graph *g = n->owningGraph();
  Node *list_node = g->appendNode(g->createList(DynamicType::get(), fmap(value, getValueTrace)));
  n->addInput(list_node->output());
}

void addInputs(Node* n, const char * name, const at::TensorOptions& options) {
  // [TensorOptions in script] - update this when you change how we schematize TensorOptions
  detail::genericAddInput(n, static_cast<int64_t>(options.dtype()));
  detail::genericAddInput(n, static_cast<int64_t>(options.layout()));
  std::vector<int64_t> device = {
      static_cast<int64_t>(options.device().type()),
      static_cast<int64_t>(options.device().index())};
  detail::genericAddInput(n, std::move(device));
}

void addInputs(Node *n, const char * name, at::IntList value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  std::vector<Value*> info = ArgumentStash::hasIntList(name) ?
    ArgumentStash::popIntList(name) :
    ArgumentStash::IntListTrace(value.size());

  auto& g = getTracingState()->graph;
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i] != nullptr) continue;
    info[i] = g->insertConstant(value[i]);
    recordSourceLocation(info[i]->node());
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

void addInputs(Node *n, const char * name, const ArrayRef<double>& value) {
  AT_ERROR("Tracing float lists currently not supported!");
}

void addOutput(Node* node, const at::Tensor& output) {
  Value * value = node->addOutput();
  if (output.defined()) {
    value->inferTypeFrom(output);
    setValueTrace(autograd::as_variable_ref(output), value);
  }
}

void addOutput(Node* node, const std::vector<at::Tensor>& outputs) {
  Value * value = node->addOutput()->setType(ListType::ofTensors());
  Graph * graph = node->owningGraph();
  Node * unpack_node = graph->appendNode(graph->create(prim::ListUnpack, {value}, outputs.size()));
  for (size_t i = 0; i < outputs.size(); ++i) {
    Value * output_val = unpack_node->outputs()[i];
    output_val->inferTypeFrom(outputs[i]);
    setValueTrace(outputs[i], output_val);
  }
}

const std::shared_ptr<TracingState>& getTracingState() {
  return detail::tracing_state;
}

void setTracingState(std::shared_ptr<TracingState> state) {
  detail::tracing_state = std::move(state);
}

TracingState::TracingState()
    : graph(new Graph()) {}

TracingState::~TracingState() = default;

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  auto & tracing_state = getTracingState();
  auto & graph = tracing_state->graph;

  auto size_var = autograd::make_variable(at::Scalar(var.size(dim)).toTensor());
  auto* value = getValueTrace(var);
  WithInsertPoint ipoint { graph->block() };
  auto dim_val = graph->insertConstant(dim);
  recordSourceLocation(dim_val->node());
  auto* node = graph->insertNode(graph->create(aten::size, {value, dim_val}));
  recordSourceLocation(node);
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

void defaultWarn(const std::string& str) { AT_WARN(str); }
std::atomic<warn_fn_type> warn_callback { defaultWarn };

void _do_warn(const char * _reason) {
  std::string reason { _reason };
  std::ostringstream s;
  s << std::string(reason);
  s << " might cause the trace to be incorrect. We can't record the data flow of "
       " Python values, which means the trace might not generalize to other inputs.";
  warn_callback.load()(s.str());
}

void setWarn(warn_fn_type fn) {
  warn_callback.store(fn);
}

void ensureUnique(const char * name, const at::Tensor& tensor) {
  auto aliases = tensor.storage().use_count();
  if (aliases > 1) {
    std::stringstream ss;
    ss << "There are " << aliases
       << " live references to the tensor being modified when tracing in-place operator "
       << name << " which ";
    warn(ss.str().c_str());
  }
}

}}}
