#include <torch/csrc/jit/frontend/tracer.h>

#include <ATen/Backtrace.h>
#include <ATen/ScalarOps.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Dict.h>
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/custom_class.h>

#include <memory>
#include <sstream>
#include <string>

namespace torch::jit::tracer {

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
void genericAddOptionalInput(
    Node* n,
    const char* name,
    const c10::optional<T>& value) {
  if (value) {
    jit::tracer::addInputs(n, name, *value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
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

static std::atomic<bool> tracer_state_warn_mode{true};

std::atomic<bool>& getTracerStateWarnMode() {
  return tracer_state_warn_mode;
}

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
  for (const auto i : c10::irange(env_stack.size())) {
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
static Value* getOptTensorValueTrace(const c10::optional<at::Tensor>& var) {
  return getValueTrace(IValue(var));
}
Value* TracingState::getValue(const IValue& var) {
  // allow tracing of tuples passed to List[Tensor] or Tuple[Tensor...]
  // arguments
  if (var.isTensorList()) {
    return graph
        ->insertNode(graph->createList(
            TensorType::get(),
            fmap(
                var.toTensorVector(),
                [&](const IValue& val) { return getValue(val); })))
        ->output();
  } else if (var.isTuple()) {
    return graph
        ->insertNode(graph->createTuple(fmap(
            var.toTupleRef().elements(),
            [&](const IValue& val) { return getValue(val); })))
        ->output();
  } else if (var.isGenericDict()) {
    auto dict = var.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();
    std::vector<Value*> keys;
    std::vector<Value*> values;
    for (const auto& entry : dict) {
      keys.emplace_back(getValue(entry.key()));
      values.emplace_back(getValue(entry.value()));
    }
    auto dict_node = graph->createDict(key_type, value_type, keys, values);
    return graph->insertNode(dict_node)->output();
  }
  if (var.isTensor()) {
    auto& ten = var.toTensor();
    if (!ten.defined()) {
      Node* n = graph->createNone();
      return graph->insertNode(n)->output();
    }
    for (const auto i : c10::irange(env_stack.size())) {
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
    if (ten.requires_grad()) {
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
    for (const auto i : c10::irange(env_stack.size())) {
      auto& future_map = env_stack.at(env_stack.size() - 1 - i);
      auto it = future_map.find(var);
      if (it == future_map.end()) {
        continue;
      }
      return it->second;
    }

    // Find torchbind classes
    if (isCustomClass(var)) {
      auto obj = Object(var.toObject());
      auto qualname = obj.type()->name();
      auto custom_class_type = getCustomClass(qualname->qualifiedName());
      if (custom_class_type) {
        auto capsule = var.toObject()->getAttr("capsule");
        for (const auto i : c10::irange(env_stack.size())) {
          auto& value_map = env_stack.at(env_stack.size() - 1 - i);
          auto it = value_map.find(capsule);
          if (it == value_map.end()) {
            continue;
          }
          return it->second;
        }
      }
    }

    std::ostringstream oss;
    if (var.isFuture()) {
      oss << "Tried to trace Future or Object that the tracer was not aware of.";
    } else {
      oss << "Tried to trace " << var
          << " but it is not part of the active trace. Modules that are called during a trace"
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
  for (const auto& frame : env_stack) {
    if (frame.count(var)) {
      return true;
    }
  }
  return false;
}

Value* TracingState::getOutput(const IValue& iv, size_t i) {
  bool tracing_mode_strict = getTracingState()->strict;
  if (iv.isTensor()) {
    const at::Tensor& var = iv.toTensor();
    if (!var.defined()) {
      Node* n = graph->createNone();
      return graph->insertNode(n)->output();
    }

    auto& value_map = getTracingState()->env_stack.back();
    auto it = value_map.find(iv);
    if (it == value_map.end()) {
      std::ostringstream os;
      os << "output " << i << " (" << var
         << ") of traced region did not have observable "
         << "data dependence with trace inputs; this probably indicates your "
            "program "
         << "cannot be understood by the tracer.";
      throw std::runtime_error(os.str());
    }
    return it->second;
  } else if (iv.isTensorList()) {
    if (tracing_mode_strict) {
      tracer::warn(
          "Encountering a list at the output of the tracer", STRICT_TRACER_MSG);
    }
    return graph
        ->insertNode(graph->createList(
            TensorType::get(),
            fmap(
                iv.toTensorVector(),
                [&](const IValue& ival) { return getOutput(ival, i); })))
        ->output();
  } else if (iv.isTuple()) {
    const auto& tuple = iv.toTupleRef().elements();
    auto tuple_node = graph->createTuple(
        fmap(tuple, [&](const IValue& ival) { return getOutput(ival, i); }));
    graph->insertNode(tuple_node);
    return tuple_node->output();
  } else if (iv.isGenericDict()) {
    if (tracing_mode_strict) {
      throw std::runtime_error(
          "Encountering a dict at the output of the tracer" +
          std::string(STRICT_TRACER_MSG));
    }
    auto dict = iv.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();

    bool key_type_valid = key_type->isSubtypeOf(*StringType::get()) ||
        key_type->isSubtypeOf(*TensorType::get());
    bool value_type_valid = value_type->isSubtypeOf(*TensorType::get());

    // Support tuple values that contain only tensors
    if (value_type->isSubtypeOf(*AnyTupleType::get())) {
      value_type_valid = true;
      for (const auto& type : value_type->containedTypes()) {
        if (!type->isSubtypeOf(*TensorType::get())) {
          value_type_valid = false;
          break;
        }
      }
    }

    if (!key_type_valid || !value_type_valid) {
      std::ostringstream os;
      os << "output " << i << " (" << dict << ") of traced region "
         << "cannot be understood by the tracer, only outputs matching"
         << "dict[Union[str, Tensor], Union[Tensor, Tuple[Tensor, ...]]] "
         << "can be a dictionary output of a traced function";
      throw std::runtime_error(os.str());
    }
    std::vector<Value*> keys;
    std::vector<Value*> values;
    for (const auto& entry : dict) {
      keys.emplace_back(getValue(entry.key()));
      values.emplace_back(getOutput(entry.value(), i));
    }
    auto dict_node = graph->createDict(key_type, value_type, keys, values);
    graph->insertNode(dict_node);
    return dict_node->output();
  } else {
    AT_ERROR(
        "Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions");
  }
}

Node* TracingState::createNode(c10::Symbol op_name, size_t num_outputs) {
  return graph->create(op_name, num_outputs);
}

void TracingState::insertNode(Node* node) {
  graph->insertNode(node);
}

// XXX: this function mutates input
static IValue addInput(
    const std::shared_ptr<TracingState>& state,
    const IValue& input,
    const TypePtr& type,
    Value* value) {
  value->setType(type);
  if (type->isSubtypeOf(*TensorType::get())) {
    auto input_tensor = input.toTensor();
    auto name = Variable(input_tensor).name();
    if (state->hasValue(input)) {
      input_tensor = input_tensor.view(input_tensor.sizes());
    }
    if (!value->hasDebugName()) {
      value->setDebugName(name);
    }
    state->setValue(input_tensor, value);
    return input_tensor;
  } else if (auto tuple_type = type->cast<TupleType>()) {
    auto unpack_node =
        state->graph->insertNode(state->graph->createTupleUnpack(value));
    auto elem_values = unpack_node->outputs();
    auto elem_types = tuple_type->elements();
    auto tuple = input.toTuple();
    const auto& elems = tuple->elements();
    size_t num_elems = elems.size();
    AT_ASSERT(
        elem_values.size() == num_elems && elem_types.size() == num_elems);
    for (const auto i : c10::irange(num_elems)) {
      tuple->unsafeSetElement(
          i, addInput(state, elems.at(i), elem_types[i], elem_values[i]));
    }
    return tuple;
  } else if (auto dict_type = type->cast<DictType>()) {
    auto dict = input.toGenericDict();

    // Unpack the list values statically
    for (const auto& entry : dict) {
      const IValue& key = entry.key();
      auto static_key = state->graph->insertConstant(key);
      auto static_value =
          state->graph->insert(aten::__getitem__, {value, static_key});
      recordSourceLocation(static_value->node());
      dict.insert_or_assign(
          entry.key(),
          addInput(
              state, entry.value(), dict_type->getValueType(), static_value));
    }

    return dict;
  } else if (auto list_type = type->cast<ListType>()) {
    size_t num_elems = input.isList() ? input.toListRef().size()
                                      : input.toTensorVector().size();
    auto list_unpack = state->graph->insertNode(
        state->graph->createListUnpack(value, num_elems));
    auto unpack_outputs = list_unpack->outputs();

    if (input.isTensorList()) {
      auto elems = input.toTensorList();
      for (const auto i : c10::irange(num_elems)) {
        elems[i] = addInput(
                       state,
                       elems.get(i),
                       list_type->getElementType(),
                       unpack_outputs[i])
                       .toTensor();
      }
      return elems;
    } else {
      auto elems = input.toList();
      for (const auto i : c10::irange(num_elems)) {
        elems[i] = addInput(
            state,
            elems.get(i),
            list_type->getElementType(),
            unpack_outputs[i]);
      }
      return elems;
    }
  } else {
    AT_ERROR(
        "Only tensors or (possibly nested) dict or tuples of tensors can be "
        "inputs to traced functions. Got ",
        type->repr_str());
  }
}

static void gatherParametersAndBuffers(
    const std::shared_ptr<TracingState>& state,
    Value* self_value,
    const Module& self,
    const std::string& prefix) {
  Graph& g = *self_value->owningGraph();

  state->setValue(self._ivalue(), self_value);

  auto self_ty = self.type();
  for (const NameValue& s : self.named_attributes(/*recurse=*/false)) {
    auto qualname = prefix + "." + s.name;
    Value* trace_get_attr = g.insertNode(g.create(prim::TracedAttr))
                                ->s_(attr::scope, qualname)
                                ->output()
                                ->setType(s.value.type());
    if (s.value.type()->isSubtypeOf(*TensorType::get())) {
      addInput(state, s.value, s.value.type(), trace_get_attr);
    }
    if (isCustomClass(s.value)) {
      tracer::setValueTrace(s.value, trace_get_attr);
    }

    auto attr_type = self_ty->getAttribute(s.name);
    // Skipping Parameters and Buffers that are behind an `InterfaceType`
    // because it is illegal for InterfaceType to expose any attribute.
    // And these attributes should never be used/exposed outside of
    // InterfaceType'd module anyway.
    if (attr_type->is_module() &&
        attr_type->kind() != TypeKind::InterfaceType) {
      gatherParametersAndBuffers(
          state, trace_get_attr, Module(s.value.toObject()), qualname);
    }
  }
}

std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self,
    const std::vector<std::string>& argument_names) {
  try {
    // Start tracing, treating 'inputs' as inputs to the trace, which can be
    // varied on subsequent invocations of the trace.  Any other variables
    // will be treated as constants.
    if (isTracing()) {
      AT_ERROR("Tracing can't be nested");
    }
    auto state = std::make_shared<TracingState>();
    setTracingState(state);

    // if we are a module, then make sure the modules parameters are in the map
    // and mapped to accesses to the self object
    if (self) {
      Value* self_value = state->graph->insertInput(0, "self")->setType(
          self->_ivalue()->type());
      gatherParametersAndBuffers(state, self_value, *self, {"__module"});
    }

    // When enough argument name hints are provided, use them as debug names
    // for traced function/modules.
    // Here argument_names is allowed to have more names than needed because
    // some arguments may have valid default values, therefore they don't need
    // example inputs.
    if (argument_names.size() >= inputs.size()) {
      for (size_t i = 0, e = inputs.size(); i < e; ++i) {
        IValue& input = inputs[i];
        input = addInput(
            state,
            input,
            input.type(),
            state->graph->addInput(argument_names[i]));
      }
    } else {
      for (IValue& input : inputs) {
        input = addInput(state, input, input.type(), state->graph->addInput());
      }
    }

    auto graph = state->graph;

    getTracingState()->lookup_var_name_fn = std::move(var_name_lookup_fn);
    getTracingState()->strict = strict;
    getTracingState()->force_outplace = force_outplace;

    // Invoke the traced function
    auto out_stack = traced_fn(inputs);

    // Exit a trace, treating 'out_stack' as the outputs of the trace.  These
    // are the variables whose values will be computed upon subsequent
    // invocations of the trace.
    size_t i = 0;
    for (auto& output : out_stack) {
      // NB: The stack is in "reverse" order, so when we pass the diagnostic
      // number we need to flip it based on size.
      state->graph->registerOutput(
          state->getOutput(output, out_stack.size() - i));
      i++;
    }
    setTracingState(nullptr);

    if (getInlineEverythingMode()) {
      Inline(*graph);
    }
    FixupTraceScopeBlocks(graph, self);
    NormalizeOps(graph);
    return {state, out_stack};
  } catch (...) {
    tracer::abandon();
    throw;
  }
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
    auto& var = v.toTensor();
    AT_ASSERT(var.defined());
    env_stack.back()[v] = value;

    // If the value comes from a CallFunction or CallMethod, it may not have
    // shape information attached. For debuggability, we enhance the type
    // information by assigning the concrete value's tupe to the jit::Value.
    if (auto tensor_type = value->type()->cast<TensorType>()) {
      if (!tensor_type->isComplete()) {
        value->inferTypeFrom(var);
      }
    }
  } else if (v.isTensorList()) {
    auto outputs = v.toTensorList();
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, outputs.size()));
    for (const auto i : c10::irange(outputs.size())) {
      setValue(outputs.get(i), unpack_node->outputs()[i]);
    }
  } else if (v.isTuple()) {
    const auto& outputs = v.toTupleRef().elements();
    Node* unpack_node = graph->insertNode(graph->createTupleUnpack(value));
    for (const auto i : c10::irange(outputs.size())) {
      setValue(outputs[i], unpack_node->outputs()[i]);
    }
  } else if (v.isList()) {
    auto elements = v.toListRef();
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, elements.size()));
    for (const auto i : c10::irange(elements.size())) {
      setValue(elements[i], unpack_node->outputs()[i]);
    }
  } else if (isCustomClass(v)) {
    auto capsule = v.toObject()->getAttr("capsule");
    env_stack.back()[capsule] = value;
  } else if (v.isFuture() || v.isObject()) {
    env_stack.back()[v] = value;
  } else if (v.isGenericDict()) {
    auto dict = v.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();
    for (const auto& entry : dict) {
      auto static_key = graph->insertConstant(entry.key());
      auto static_value = graph->insert(aten::__getitem__, {value, static_key});
      setValue(entry.value(), static_value);
    }
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

void addInputs(Node* n, const char* name, c10::SymInt value) {
  addInputs(n, name, value.expect_int());
}

void addInputs(Node* n, const char* name, c10::optional<int64_t> value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  if (ArgumentStash::hasValue(name)) {
    Value* v = ArgumentStash::popValue(name);
    n->addInput(v);
  } else if (value) {
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
void addInputs(Node* n, const char* name, const c10::optional<bool>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(Node* n, const char* name, double value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name, const c10::optional<double>& value) {
  detail::genericAddOptionalInput(n, name, value);
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
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(Node* n, const char* name, const c10::string_view value) {
  detail::genericAddInput(n, std::string(value));
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<c10::string_view>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(Node* n, const char* name, const at::Tensor& value) {
  n->addInput(getValueTrace(value));
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Tensor>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Generator>& value) {
  if (value.has_value() && value->defined()) {
    detail::badArgType(*value);
  }
  Graph* g = n->owningGraph();
  Value* undef_gen = g->insertNode(g->createNone())->output();
  n->addInput(undef_gen);
}
void addInputs(Node* n, const char* name, at::Device value) {
  detail::genericAddInput(n, value);
}
void addInputs(Node* n, const char* name, c10::Stream stream) {
  detail::genericAddInput(n, c10::IValue(stream));
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
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Layout>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Device>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(
    Node* n,
    const char* name,
    c10::optional<at::DimnameList> value) {
  TORCH_CHECK(false, "NYI: Named tensors are not supported with the tracer");
}
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::ScalarType>& value) {
  detail::genericAddOptionalInput(n, name, value);
}
void addInputs(
    Node* n,
    const char* name,
    at::ArrayRef<at::Tensor> value,
    bool allow_undefined) {
  addInputs(n, name, at::ITensorListRef(value), allow_undefined);
}
void addInputs(
    Node* n,
    const char* name,
    std::vector<at::Tensor> value,
    bool allow_undefined) {
  addInputs(n, name, at::ITensorListRef(value), allow_undefined);
}
void addInputs(
    Node* n,
    const char* name,
    at::ITensorListRef value,
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
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const List<c10::optional<at::Tensor>>& value) {
  Graph* g = n->owningGraph();
  Node* list_node = nullptr;
  list_node = g->insertNode(g->createList(
      OptionalType::ofTensor(), fmap(value, getOptTensorValueTrace)));
  n->addInput(list_node->output());
}
void addInputs(
    Node* n,
    const char* name,
    ArrayRef<c10::intrusive_ptr<c10::ivalue::Object>> value,
    const ClassTypePtr& class_type) {
  Graph* g = n->owningGraph();
  Node* list_node =
      g->insertNode(g->createList(class_type, fmap(value, getValueTrace)));
  n->addInput(list_node->output());
}

void addInputs(Node* n, const char* name, at::IntArrayRef value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  std::vector<Value*> info = ArgumentStash::hasIntArrayRef(name)
      ? ArgumentStash::popIntArrayRef(name)
      : ArgumentStash::IntArrayRefTrace(value.size());

  auto& g = getTracingState()->graph;
  for (const auto i : c10::irange(info.size())) {
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

void addInputs(Node* n, const char* name, c10::SymIntArrayRef value) {
  addInputs(n, name, C10_AS_INTARRAYREF_SLOW(value));
}

void addInputs(Node* n, const char* name, c10::optional<c10::SymInt> value) {
  addInputs(
      n,
      name,
      value.has_value() ? c10::make_optional(value->expect_int())
                        : c10::nullopt);
}

void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::IntArrayRef>& opt_value) {
  detail::genericAddOptionalInput(n, name, opt_value);
}

void addInputs(
    Node* n,
    const char* name,
    const at::OptionalIntArrayRef& opt_value) {
  if (opt_value.has_value()) {
    jit::tracer::addInputs(n, name, *opt_value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}

void addInputs(
    Node* n,
    const char* name,
    const at::OptionalSymIntArrayRef& opt_value) {
  if (opt_value.has_value()) {
    jit::tracer::addInputs(n, name, *opt_value);
  } else {
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}

void addInputs(Node* n, const char* name, ArrayRef<double> value) {
  std::vector<Value*> info;
  auto& g = getTracingState()->graph;
  for (double elt : value) {
    info.push_back(g->insertConstant(elt));
    recordSourceLocation(info.back()->node());
  }
  n->addInput(
      g->insertNode(g->createList(jit::FloatType::get(), info))->output());
}

void addInputs(
    Node* n,
    const char* name,
    const c10::optional<c10::ArrayRef<double>>& opt_value) {
  detail::genericAddOptionalInput(n, name, opt_value);
}

void addInputs(
    Node* n,
    const char* name,
    const c10::intrusive_ptr<c10::ivalue::Object>& obj) {
  Value* v = getValueTrace(obj);
  n->addInput(v);
}

void addOutput(Node* node, const at::Tensor& output) {
  setOutput(node->addOutput(), output);
}

void setOutput(Value* value, const at::Tensor& output) {
  if (output.defined()) {
    value->inferTypeFrom(output);
    setValueTrace(output, value);
  }
}

void addOutput(Node* node, const std::vector<at::Tensor>& outputs) {
  Value* value = node->addOutput()->setType(ListType::ofTensors());
  Graph* graph = node->owningGraph();
  Node* unpack_node = graph->insertNode(
      graph->create(prim::ListUnpack, {value}, outputs.size()));
  for (const auto i : c10::irange(outputs.size())) {
    Value* output_val = unpack_node->outputs()[i];
    output_val->inferTypeFrom(outputs[i]);
    setValueTrace(outputs[i], output_val);
  }
}

void addOutput(Node* node, const c10::List<at::Tensor>& outputs) {
  return addOutput(node, outputs.vec());
}

void addOutput(
    Node* node,
    const c10::intrusive_ptr<c10::ivalue::Object>& output) {
  Value* output_val = node->addOutput();
  output_val->inferTypeFrom(output);
  setValueTrace(output, output_val);
}

const std::shared_ptr<TracingState>& getTracingState() {
  return detail::tracing_state;
}

void setTracingState(std::shared_ptr<TracingState> state) {
  at::tracer::impl::set_dispatch_enabled(state != nullptr);
  detail::tracing_state = std::move(state);
}

TracingState::TracingState() : graph(new Graph()), env_stack{Frame()} {}

TracingState::~TracingState() = default;

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  auto& tracing_state = getTracingState();
  auto& graph = tracing_state->graph;

  Variable size_var;
  {
    // Make sure this scalar to tensor isn't traced!
    at::AutoDispatchBelowADInplaceOrView guard;
    size_var = scalar_to_tensor(at::Scalar(var.size(dim)));
  }
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

autograd::Variable getNumelOf(const autograd::Variable& var) {
  auto& tracing_state = getTracingState();
  auto& graph = tracing_state->graph;

  Variable numel_var;
  {
    // Make sure this scalar to tensor isn't traced!
    at::AutoDispatchBelowADInplaceOrView guard;
    numel_var = scalar_to_tensor(at::Scalar(var.numel()));
  }
  auto* value = getValueTrace(var);
  auto* node = graph->insertNode(graph->create(Symbol::aten("numel"), {value}));
  recordSourceLocation(node);
  node->output()->setType(jit::IntType::get());

  auto ten =
      graph->insertNode(graph->createNumToTensor(node->output()))->output();
  setValueTrace(numel_var, ten);
  return numel_var;
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
void ensureUniqueIfOutOfPlaced(
    const char* name,
    const c10::optional<at::Tensor>& tensor) {
  ensureUniqueIfOutOfPlaced(name, tensor.has_value() ? *tensor : at::Tensor());
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
  IntArrayRefTrace& list_trace =
      stash.intlists.emplace(arg_name, size).first->second;
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
    ten = g.insert(aten::ScalarImplicit, {ten});
  }

  stash.values.emplace(arg_name, ten);
}

////////////////////////////////////////////////////////////////////////////////
// Stack trace recording
////////////////////////////////////////////////////////////////////////////////
// no python present so we just do not record source information
static void defaultRecordSourceLocation(Node* n) {}
std::atomic<decltype(&defaultRecordSourceLocation)> record_source_location(
    defaultRecordSourceLocation);
void recordSourceLocation(Node* n) {
  return record_source_location.load()(n);
}
void setRecordSourceLocation(void (*v)(Node*)) {
  record_source_location.store(v);
}

static std::vector<StackEntry> defaultPythonCallstack() {
  return std::vector<StackEntry>();
}
std::atomic<decltype(&defaultPythonCallstack)> python_callstack_fn(
    defaultPythonCallstack);
std::vector<StackEntry> pythonCallstack() {
  return python_callstack_fn.load()();
}
void setPythonCallstack(std::vector<StackEntry> (*v)()) {
  python_callstack_fn.store(v);
}

static void defaultWarn(const std::string& str) {
  TORCH_WARN(str);
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
    "up as a constant in the graph. Consider using `view` or `reshape` to make "
    "it traceable.";
const char* STRICT_TRACER_MSG =
    " might cause the trace to be incorrect, this is only valid if the container "
    "structure does not change based on the module's inputs. Consider using a constant "
    "container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a "
    "`NamedTuple` instead). If you absolutely need this and know the side effects, pass "
    "strict=False to trace() to allow this behavior.";
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
} // namespace torch::jit::tracer
