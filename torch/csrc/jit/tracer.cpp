#include <torch/csrc/jit/tracer.h>

#include <torch/csrc/utils/variadic.h>
#include <torch/csrc/jit/constants.h>
#include <ATen/core/functional.h>
#include <ATen/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/script/module.h>
#include <ATen/core/Dict.h>
#include <ATen/core/EnableNamedTensor.h>

#include <memory>
#include <sstream>
#include <string>

namespace torch {
namespace jit {
namespace tracer {

////////////////////////////////////////////////////////////////////////////////
// Recording the traces
////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <typename T>
void genericAddInput(Node* n, T value) {
  Value* v = n->owningGraph()->insertConstant(value);
  recordSourceLocation(v->node());
  n->addInput(v);
}

template <typename T>
void badArgType(const T& v) {
  AT_ERROR(
      "Found an unsupported argument type in the JIT tracer: ",
      c10::demangle_type<T>(),
      ". File a bug report.");
}

thread_local std::shared_ptr<TracingState> tracing_state;
} // namespace detail

std::function<void()> pauseTracing() {
  // NOLINTNEXTLINE
  std::shared_ptr<tracer::TracingState> state = getTracingState();
  tracer::setTracingState(nullptr);

  return [state]() { tracer::setTracingState(state); };
}

void delValueTrace(const IValue& var) {
  getTracingState()->delValue(var);
}
void TracingState::delValue(const IValue& var) {
  for (size_t i = 0; i < env_stack.size(); ++i) {
    auto& value_map = env_stack.at(env_stack.size() - 1 - i);
    auto it = value_map.find(var);
    if (it == value_map.end()) {
      continue;
    }
    value_map.erase(it);
  }
}

// Given a IValue 'var', return the 'node' which represents the instruction
// which computes the value of this variable in the IR.
// Here, we interpret untraced variables as constants that are just embedded
// in the graph.  This is useful to handle code which does things like this
// (from torch.autograd.variable, now moved to C++):
//
//    def mm(self, matrix):
//      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
//      return Addmm.apply(output, self, matrix, 0, 1, True)
//
// Here, mm fakes up a dummy variable with uninitialized data to do an inplace
// update on, but subsequently ignores it because the alpha scaling factor is
// zero. This is one of the cases where a Variable can be created inside of a
// trace, and if we treat it as a constant, everything will work out.
Value* getValueTrace(const IValue& var) {
  return getTracingState()->getValue(var);
}
Value* TracingState::getValue(const IValue& var) {
  // allow tracing of tuples passed to List[Tensor] or Tuple[Tensor...] arguments
  if (var.isTensorList()) {
    return graph
        ->insertNode(graph->createList(
            TensorType::get(),
            fmap(
                var.toTensorListRef(),
                [&](const IValue& val) { return getValue(val); })))
        ->output();
  } else if (var.isTuple()) {
    return graph
        ->insertNode(graph->createTuple(fmap(
            var.toTuple()->elements(),
            [&](const IValue& val) { return getValue(val); })))
        ->output();
  } if (var.isTensor()) {
    auto ten = var.toTensor();
    if (!ten.defined()) {
      Node* n = graph->createNone();
      return graph->insertNode(n)->output();
    }
    for (size_t i = 0; i < env_stack.size(); ++i) {
      auto& value_map = env_stack.at(env_stack.size() - 1 - i);
      auto it = value_map.find(var);
      if (it == value_map.end()) {
        continue;
      }
      if (!it->second->hasDebugName()) {
        auto unique_name = getTracingState()->lookup_var_name_fn(ten);
        if (!unique_name.empty()) {
          it->second->setDebugName(unique_name);
        }
      }
      return it->second;
    }

    // Didn't find it. Bake in a constant
    if (ten.is_variable() && ten.requires_grad()) {
      pauseTracing();
      std::ostringstream oss;
      oss << "Cannot insert a Tensor that requires grad as a constant. "
          << "Consider making it a parameter or input, or detaching the gradient\n"
          << "Tensor:\n"
          << ten;
      throw std::runtime_error(oss.str());
    }

    Value* constant = graph->insertConstant(ten);
    recordSourceLocation(constant->node());
    constant->inferTypeFrom(ten);
    auto it = env_stack.back().emplace(var, constant);
    return it.first->second;
  } else if (var.isFuture() || var.isObject()) {
    for (size_t i = 0; i < env_stack.size(); ++i) {
      auto& future_map = env_stack.at(env_stack.size() - 1 - i);
      auto it = future_map.find(var);
      if (it == future_map.end()) {
        continue;
      }
      return it->second;
    }
    std::ostringstream oss; 
    if (var.isFuture()) {
      oss << "Tried to trace Future or Object that the tracer was not aware of.";
    } else {
      oss << "Tried to trace " << var << " but it is not part of the active trace. Modules that are called during a trace"
      << " must be registered as submodules of the thing being traced.";
    }
    throw std::runtime_error(oss.str());
  } else {
    // If the values are non-tensors, we try to create constants
    // and bake those constants into the traced graph
    auto constant = tryInsertConstant(*graph, var);
    if (constant) {
      recordSourceLocation(constant.value()->node());
      return *constant;
    }
    std::ostringstream os;
    os << "Tracer cannot get value trace for type " << var.tagKind() << ". "
       << "The below value could not be materialized as a constant:\n"
       << var;
    throw std::runtime_error(os.str());
  }
}
bool TracingState::hasValue(const IValue& var) const {
  for(const auto & frame : env_stack) {
    if (frame.count(var)) {
      return true;
    }
  }
  return false;
}


Value* TracingState::getOutput(const IValue& iv) {
   if (iv.isTensor()) {
     at::Tensor var = iv.toTensor();
     if (!var.defined()) {
       Node* n = graph->createNone();
       return graph->insertNode(n)->output();
     }

     auto &value_map = getTracingState()->env_stack.back();
     auto it = value_map.find(iv);
     if (it == value_map.end()) {
       std::ostringstream os;
       os << "output of traced region did not have observable "
          << "data dependence with trace inputs; this probably indicates your "
             "program "
          << "cannot be understood by the tracer.";
       throw std::runtime_error(os.str());
     }
     return it->second;
  } else if (iv.isTuple()) {
    auto tuple = iv.toTuple()->elements();
    auto tuple_node = graph->createTuple(
        fmap(tuple, [&](const IValue& ival) { return getOutput(ival); }));
    graph->insertNode(tuple_node);
    return tuple_node->output();
  } else {
    AT_ERROR(
        "Only tensors or tuples of tensors can be output from traced functions");
  }
}

// XXX: this function mutates input
static IValue addInput(const std::shared_ptr<TracingState> & state, const IValue& input, const TypePtr& type, Value* value) {
  value->setType(type);
  if (type->isSubtypeOf(TensorType::get())) {
    auto input_tensor = input.toTensor();
    auto name = Variable(input_tensor).name();
    if (state->hasValue(input)) {
      input_tensor = input_tensor.view(input_tensor.sizes());
    }
    value->setDebugName(name);
    state->setValue(input_tensor, value);
    return input_tensor;
  } else if (auto tuple_type = type->cast<TupleType>()) {
    auto unpack_node =
        state->graph->insertNode(state->graph->createTupleUnpack(value));
    auto elem_values = unpack_node->outputs();
    auto elem_types = tuple_type->elements();
    auto tuple = input.toTuple();
    auto elems = tuple->elements();
    size_t num_elems = elems.size();
    AT_ASSERT(
        elem_values.size() == num_elems && elem_types.size() == num_elems);
    for (size_t i = 0; i < num_elems; ++i) {
      elems[i] = addInput(state, elems.at(i), elem_types[i], elem_values[i]);
    }
    return std::move(tuple);
  } else if (auto dict_type = type->cast<DictType>()) {
    auto dict = input.toGenericDict();

    auto dict_size = dict.size();
    auto unpack_to_list = state->graph->insert(aten::values, {value});
    auto list_unpack = state->graph->createListUnpack(unpack_to_list, dict_size);
    auto unpack_node = state->graph->insertNode(list_unpack);
    auto elem_values = unpack_node->outputs();

    const auto order = iterationOrder(dict);
    AT_ASSERT(order.size() == elem_values.size());

    size_t i = 0;
    for (const auto &pair : order) {
      dict.insert_or_assign(pair.first, addInput(state, pair.second, dict_type->getValueType(), elem_values[i++]));
    }

    return std::move(dict);
  } else if (auto list_type = type->cast<ListType>()) {
    size_t num_elems = input.isGenericList() ? input.toGenericListRef().size()
                                             : input.toTensorListRef().size();
    auto list_unpack = state->graph->insertNode(state->graph->createListUnpack(value, num_elems));
    auto unpack_outputs = list_unpack->outputs();

    if (input.isTensorList()) {
      auto elems = input.toTensorList();
      for (size_t i = 0; i < num_elems; i++) {
        elems[i] = addInput(state, elems.get(i), list_type->getElementType(), unpack_outputs[i]).toTensor();
      }
      return elems;
    } else {
      auto elems = input.toGenericList();
      for (size_t i = 0; i < num_elems; i++) {
        elems[i] = addInput(state, elems.get(i), list_type->getElementType(), unpack_outputs[i]);
      }
      return elems;
    }
  } else {
    AT_ERROR(
        "Only tensors or (possibly nested) dict or tuples of tensors can be "
        "inputs to traced functions. Got ", type->python_str());
  }
}

static void gatherParametersAndBuffers(
    const std::shared_ptr<TracingState>& state,
    Value* self_value,
    const script::Module& self) {
  Graph& g = *self_value->owningGraph();
  
  state->setValue(self.module_object(), self_value);

  for (script::Slot s : self.get_slots()) {
    if (s.type()->isSubtypeOf(TensorType::get())) {
      addInput(
          state, s.value(), s.type(), g.insertGetAttr(self_value, s.name()));
    } else if (s.entity_type() == script::EntityType::MODULE) {
      gatherParametersAndBuffers(
          state, g.insertGetAttr(self_value, s.name()), s.to_module());
    }
  }
}


// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
std::pair<std::shared_ptr<TracingState>, Stack> enter(
    TypedStack inputs,
    script::Module* self) {
  if (isTracing()) {
    AT_ERROR("Tracing can't be nested");
  }
  auto state = std::make_shared<TracingState>();
  setTracingState(state);

  // if we are a module, then make sure the modules parameters are in the map
  // and mapped to accesses to the self object
  if (self) {
    Value* self_value =
        state->graph->insertInput(0, "self")->setType(self->module_object()->type());
    gatherParametersAndBuffers(state, self_value, *self);
  }

  size_t i = 0;
  auto input_types = inputs.types()->elements();
  for (IValue& input : inputs.stack()) {
    input = addInput(state,
        input, input_types[i++], state->graph->addInput());
  }
  return std::make_pair(state, inputs.stack());
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
void exit(const Stack& outputs) {
  auto& state = getTracingState();
  size_t i = 0;
  for (auto& output : outputs) {
    state->graph->registerOutput(state->getOutput(output));
    i++;
  }
  setTracingState(nullptr);
}

// Abort tracing. Used to reset the state in case of errors.
void abandon() {
  setTracingState(nullptr);
}

void setValueTrace(const IValue& v, Value* value) {
  return getTracingState()->setValue(v, value);
}
void TracingState::setValue(const IValue& v, Value* value) {
  if (v.isTensor()) {
    auto var = v.toTensor();
    AT_ASSERT(var.defined());
    env_stack.back()[v] = value;
  } else if (v.isTensorList()) {
    auto outputs = v.toTensorList();
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, outputs.size()));
    for (size_t i = 0; i < outputs.size(); ++i) {
      setValue(outputs.get(i), unpack_node->outputs()[i]);
    }
  } else if (v.isTuple()) {
    auto outputs = v.toTuple()->elements();
    Node* unpack_node = graph->insertNode(graph->createTupleUnpack(value));
    for (size_t i = 0; i < outputs.size(); ++i) {
      setValue(outputs[i], unpack_node->outputs()[i]);
    }
  } else if (v.isGenericList()) {
    auto elements = v.toGenericListRef();
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, elements.size()));
    for (size_t i = 0; i < elements.size(); ++i) {
      setValue(elements[i], unpack_node->outputs()[i]);
    }
  } else if (v.isFuture() || v.isObject()) {
    env_stack.back()[v] = value;
  } else {
    std::ostringstream os;
    os << "Tracer cannot set value trace for type " << v.tagKind() << ". "
       << "Supported types are tensor, tensor list, and tuple of tensors.";
    throw std::runtime_error(os.str());
  }
}

void addInputs(Node* n, const char* name, int64_t value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  if (ArgumentStash::hasValue(name)) {
    Value* v = ArgumentStash::popValue(name);
    n->addInput(v);
  } else {
    detail::genericAddInput(n, value);
  }
}

void addInputs(Node* n, const char* name, c10::optional<int64_t> value) {
  if (value) {
    detail::genericAddInput(n, *value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}
void addInputs(Node* n, const char* name, bool value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name /* unused */, const c10::optional<bool>& value) {
  if (value) {
    detail::genericAddInput(n, *value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}
void addInputs(Node* n, const char* name, double value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name, const at::Scalar& value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  if (ArgumentStash::hasValue(name)) {
    Value* v = ArgumentStash::popValue(name);
    n->addInput(v);
  } else {
    detail::genericAddInput(n, value);
  }
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Scalar>& value) {
  if (value) {
    detail::genericAddInput(n, *value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}
void addInputs(Node* n, const char* name, const std::string& value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name, const at::Tensor& value) {
  n->addInput(getValueTrace(value));
}
void addInputs(Node* n, const char* name, at::Generator* value) {
  if (value) {
    detail::badArgType(value);
  }
  Graph* g = n->owningGraph();
  Value* undef_gen = g->insertNode(g->createNone())->output();
  n->addInput(undef_gen);
}
void addInputs(Node* n, const char* name, at::Device value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name, at::Layout value) {
  detail::genericAddInput(n, static_cast<int64_t>(value));
}
void addInputs(Node* n, const char* name, at::ScalarType value) {
  detail::genericAddInput(n, static_cast<int64_t>(value));
}
void addInputs(Node* n, const char* name, at::MemoryFormat value) {
  detail::genericAddInput(n, static_cast<int64_t>(value));
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::MemoryFormat>& value) {
  if (value) {
    detail::genericAddInput(n, static_cast<int64_t>(*value));
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}
#ifdef BUILD_NAMEDTENSOR
void addInputs(
    Node* n,
    const char* name,
    c10::optional<at::DimnameList> value) {
  TORCH_CHECK(false, "NYI: Named tensors are not supported with the tracer");
}
#endif
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::ScalarType>& value) {
  if (value.has_value()) {
    detail::genericAddInput(n, static_cast<int64_t>(*value));
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}

void addInputs(
    Node* n,
    const char* name,
    at::TensorList value,
    bool allow_undefined) {
  Graph* g = n->owningGraph();
  Node* list_node = nullptr;
  if (allow_undefined) {
    // if allow undefined, we create a list of optional tensors
    list_node = g->insertNode(
        g->createList(OptionalType::ofTensor(), fmap(value, getValueTrace)));
  } else {
    list_node = g->insertNode(
        g->createList(TensorType::get(), fmap(value, getValueTrace)));
  }
  n->addInput(list_node->output());
}

void addInputs(Node* n, const char* name, const at::TensorOptions& options) {
  // [TensorOptions in script] - update this when you change how we schematize
  // TensorOptions
  addInputs(n, name, at::typeMetaToScalarType(options.dtype()));
  addInputs(n, name, options.layout());
  addInputs(n, name, options.device());
  addInputs(n, name, options.pinned_memory());
}

void addInputs(Node* n, const char* name, at::IntArrayRef value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  std::vector<Value*> info = ArgumentStash::hasIntArrayRef(name)
      ? ArgumentStash::popIntArrayRef(name)
      : ArgumentStash::IntArrayRefTrace(value.size());

  auto& g = getTracingState()->graph;
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i] != nullptr)
      continue;
    info[i] = g->insertConstant(value[i]);
    recordSourceLocation(info[i]->node());
  }
  for (jit::Value* v : info) {
    if (*v->type() != *jit::IntType::get()) {
      throw std::runtime_error(
          "Type mismatch in setposattr for IntArrayRef. Check that your program "
          "is valid without tracing, and please file a bug report if it is.");
    }
  }
  n->addInput(
      g->insertNode(g->createList(jit::IntType::get(), info))->output());
}

void addInputs(Node* n, const char* name, const ArrayRef<double>& value) {
  AT_ERROR("Tracing float lists currently not supported!");
}

void addInputs(Node* n, const char* name, const std::vector<double>& value) {
  AT_ERROR("Tracing float lists currently not supported!");
}

void addOutput(Node* node, const at::Tensor& output) {
  setOutput(node->addOutput(), output);
}

void setOutput(Value* value, const at::Tensor& output) {
  if (output.defined()) {
    value->inferTypeFrom(output);
    setValueTrace(autograd::as_variable_ref(output), value);
  }
}

void addOutput(Node* node, const std::vector<at::Tensor>& outputs) {
  Value* value = node->addOutput()->setType(ListType::ofTensors());
  Graph* graph = node->owningGraph();
  Node* unpack_node = graph->insertNode(
      graph->create(prim::ListUnpack, {value}, outputs.size()));
  for (size_t i = 0; i < outputs.size(); ++i) {
    Value* output_val = unpack_node->outputs()[i];
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
    : graph(new Graph()), env_stack{Frame()} {}

TracingState::~TracingState() = default;

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  auto& tracing_state = getTracingState();
  auto& graph = tracing_state->graph;

  auto size_var =
      autograd::make_variable(scalar_to_tensor(at::Scalar(var.size(dim))));
  auto* value = getValueTrace(var);
  auto dim_val = graph->insertConstant(dim);
  recordSourceLocation(dim_val->node());
  auto* node = graph->insertNode(graph->create(aten::size, {value, dim_val}));
  recordSourceLocation(node);
  node->output()->setType(jit::IntType::get());

  auto ten =
      graph->insertNode(graph->createNumToTensor(node->output()))->output();
  setValueTrace(size_var, ten);
  return size_var;
}

void ensureUniqueIfOutOfPlaced(const char* name, const at::Tensor& tensor) {
  auto& state = getTracingState();
  if (state && state->force_outplace == false) {
    // If we're not converting in-place ops to out-of-place, this check is
    // unnecessary
    return;
  }
  auto aliases = tensor.storage().use_count();
  if (isTracing() && aliases > 1) {
    std::stringstream ss;
    ss << "There are " << aliases
       << " live references to the data region being modified when tracing in-place operator "
       << name
       << ". This might cause the trace to be incorrect, because all other views "
       << "that also reference this data will not reflect this change in the trace! "
       << "On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. "
       << "are outputs of torch.split), this might still be safe.";
    warn(ss.str().c_str());
  }
}

////////////////////////////////////////////////////////////////////////////////
// Argument stash
////////////////////////////////////////////////////////////////////////////////
thread_local ArgumentStash ArgumentStash::stash;

void ArgumentStash::stashIntArrayRefElem(
    const std::string& arg_name,
    size_t size,
    size_t idx,
    const Variable& var) {
  // TODO: check type?
  if (!isTracing())
    return;
  auto& list_trace = stash.intlists.emplace(arg_name, size).first->second;
  AT_ASSERT(size == list_trace.size());
  AT_ASSERT(idx < list_trace.size());
  AT_ASSERT(list_trace[idx] == nullptr);

  Value* ten = getValueTrace(var);
  auto& g = *ten->owningGraph();
  WithInsertPoint guard(ten->node()->next());
  auto prim = g.insert(aten::Int, {ten});
  list_trace[idx] = prim;
}

void ArgumentStash::stashValue(
    const std::string& arg_name,
    size_t idx,
    const Variable& var,
    const TypePtr& type) {
  if (!isTracing())
    return;

  Value* ten = getValueTrace(var);
  WithInsertPoint guard(ten->node()->next());
  auto& g = *ten->owningGraph();

  if (type == IntType::get()) {
    ten = g.insert(aten::Int, {ten});
  } else if (type == FloatType::get()) {
    ten = g.insert(aten::Float, {ten});
  } else if (type == NumberType::get()) {
    ten = g.insert(prim::ImplicitTensorToNum, {ten});
  }

  stash.values.emplace(arg_name, ten);
}

////////////////////////////////////////////////////////////////////////////////
// Stack trace recording
////////////////////////////////////////////////////////////////////////////////
// no python present so we just do not record source information
void defaultRecordSourceLocation(Node* n) {}
std::atomic<decltype(&defaultRecordSourceLocation)> record_source_location(
    defaultRecordSourceLocation);
void recordSourceLocation(Node* n) {
  return record_source_location.load()(n);
}
void setRecordSourceLocation(void (*v)(Node*)) {
  record_source_location.store(v);
}

void defaultWarn(const std::string& str) {
  AT_WARN(str);
}
std::atomic<warn_fn_type> warn_callback{defaultWarn};

const char* WARN_PYTHON_DATAFLOW =
    " might cause the trace to be incorrect. We can't record the data flow of "
    "Python values, so this value will be treated as a constant in the future. "
    "This means that the trace might not generalize to other inputs!";
const char* WARN_CONSTRUCTOR =
    " results are registered as constants in the trace. You can safely ignore this "
    "warning if you use this function to create tensors out of constant variables "
    "that would be the same every time you call this function. In any other case, "
    "this might cause the trace to be incorrect.";
const char* WARN_RESIZE =
    " can't be represented in the JIT at the moment, so we won't connect any uses of "
    "this value with its current trace. If you happen to use it again, it will show "
    "up as a constant in the graph.";

// XXX: _kind can be a nullptr
void _do_warn(const char* _reason, const char* _kind) {
  std::string reason{_reason};
  std::string kind{_kind ? _kind : ""};
  std::ostringstream s;
  s << reason << kind;
  warn_callback.load()(s.str());
}

void setWarn(warn_fn_type fn) {
  warn_callback.store(fn);
}
} // namespace tracer
} // namespace jit
} // namespace torch
