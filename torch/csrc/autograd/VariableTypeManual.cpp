#include <c10/util/Optional.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/utils/memory.h>

#include <torch/csrc/utils/memory.h>

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

VariableType::VariableType(Context* context, TypeExtendedInterface* baseType)
  : TypeDefault(baseType->type_id(), /*is_variable=*/true, /*is_undefined=*/false)
  , baseType(baseType)
  , id_(context->freshTypeID()) {
  str = std::string("Variable[") + baseType->toString() + "]";
}

Backend VariableType::backend() const {
  return baseType->backend();
}
Allocator* VariableType::allocator() const {
  return baseType->allocator();
}
Device VariableType::getDeviceFromPtr(void * data) const {
  return baseType->getDeviceFromPtr(data);
}
Storage VariableType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return baseType->unsafeStorageFromTH(th_pointer, retain);
}
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return make_variable(baseType->unsafeTensorFromTH(th_pointer, retain), /*requires_grad=*/false);
}
std::unique_ptr<Generator> VariableType::generator() const {
  return baseType->generator();
}

const char * VariableType::toString() const {
  return str.c_str();
}
Type & VariableType::toBackend(Backend b) const {
  return *getVariableTypeFromBaseType(baseType->toBackend(b));
}
Type & VariableType::toScalarType(ScalarType s) const {
  return *getVariableTypeFromBaseType(baseType->toScalarType(s));
}
TypeID VariableType::ID() const {
  return static_cast<TypeID>(id_);
}

std::vector<std::unique_ptr<Type>> type_to_variable_type;

// XXX - this is not threadsafe with uses of Variables
void register_variable_type_for(TypeExtendedInterface* baseType) {
  AT_ASSERT(baseType);
  const auto base_id = static_cast<size_t>(baseType->ID());
  if(type_to_variable_type.size() <= base_id) {
    type_to_variable_type.resize(base_id + 1);
  }
  type_to_variable_type[base_id] =
      make_unique<VariableType>(&at::globalContext(), baseType);
}

struct VariableTypeRegistry {
  VariableTypeRegistry() {
    auto& context = at::globalContext();
    for (int p = 0; p < static_cast<int>(Backend::NumOptions); ++p) {
      for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); ++s) {
        auto baseType = context.getNonVariableTypeRaw(static_cast<Backend>(p), static_cast<ScalarType>(s));
        if (baseType && baseType->backend() != Backend::Undefined) {
          register_variable_type_for(baseType);
        }
      }
    }
  }
};

struct VariableHooks : public at::VariableHooksInterface {
  VariableHooks(at::VariableHooksArgs) {}
  void registerVariableTypeFor(at::LegacyTypeDispatch*, at::Backend) const override;
  at::Type& getVariableTypeFromBaseType(const at::Type&) const override;
};

// Sigh, the registry doesn't support namespaces :(
using at::RegistererVariableHooksRegistry;
using at::VariableHooksRegistry;

// WARNING: YOU MUST DO THE NEXT TWO STATIC INITIALIZERS IN THIS ORDER.
//
// If you do it in the other order, this is what can happen if
// these static initializers are called before Context is
// initialized:
//
//    - VariableHooks::registerVariableTypeFor will be activated
//      to register a variable type
//
//    - We run the constructor of VariableTypeRegistry, which
//      calls at::globalContext()
//
//    - Context is not initialized yet, so we call the constructor
//      of Context
//
//    - We register CPU types, calling VariableHooks::registerVariableTypeFor
//
//    - We register the CPU type as a variable type
//
//    - In VariableTypeRegistry, we try to register the Variable type AGAIN!!
//      Disaster.
//
static VariableTypeRegistry registry;
REGISTER_VARIABLE_HOOKS(VariableHooks)

// Pre-condition: backend/scalar_type is a valid type in the type_registry
void VariableHooks::registerVariableTypeFor(at::LegacyTypeDispatch* context, at::Backend backend) const {
  auto* baseType = context->getNonVariableTypeRaw(backend, ScalarType::Undefined);
  register_variable_type_for(static_cast<at::TypeExtendedInterface*>(baseType));
}

at::Type& VariableHooks::getVariableTypeFromBaseType(const at::Type& baseType) const {
  return *VariableType::getVariableTypeFromBaseType(baseType);
}

bool VariableType::isVariableType(const at::Type& type) {
  return type.is_variable();
}

at::TypeExtendedInterface* VariableType::getVariableTypeFromBaseType(const at::Type& baseType) {
  auto id = static_cast<size_t>(baseType.ID());
  if(id >= type_to_variable_type.size())
    return nullptr;
  return static_cast<at::TypeExtendedInterface*>(type_to_variable_type[id].get());
}

namespace {
std::vector<at::Type*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  auto& context = at::globalContext();
  std::vector<Type*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    auto baseType = context.getNonVariableTypeRaw(static_cast<Backend>(p), ScalarType::Undefined);
    if (baseType) {
      res.emplace_back(VariableType::getVariableTypeFromBaseType(*baseType));
    }
  }
  return res;
}
}

std::vector<at::Type*> VariableType::allCPUTypes() {
  return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

std::vector<at::Type*> VariableType::allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({ Backend::CUDA, Backend::SparseCUDA });
}

const Variable & VariableType::checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.dispatch_type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

Variable & VariableType::checked_cast_variable(Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.dispatch_type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

const Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor & VariableType::unpack(Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

SparseTensorRef VariableType::unpack(SparseTensorRef t, const char * name, int pos) {
  return SparseTensorRef(checked_cast_variable(t.tref, name, pos));
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> VariableType::unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    }
    if (!isVariableType(t.dispatch_type())) {
      AT_ERROR("Expected object of type Variable but found type ", t.dispatch_type().toString(), " at position #", i, " "
                    "for iterable argument #", pos, " '", name, "'");
    }
    ret[i] = static_cast<const Variable&>(t);
  }
  return ret;
}

void VariableType::backward(
    Tensor& self,
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  as_variable_ref(self).backward(gradient, keep_graph, create_graph);
}

void VariableType::set_data(Tensor & self, Tensor new_data) const {
  as_variable_ref(self).set_data(new_data);
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor & VariableType::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  jit::Value* output = nullptr;
  if(torch::jit::tracer::isTracing()) {
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();
    auto& graph = state.graph;
    if (state.force_outplace) {
      // if you have no views of self, then an in place copy is equivalent to
      // making sure we expand src to the same size as self
      jit::Node* node = graph->create(jit::aten::expand_as, /*num_outputs=*/1);
      jit::tracer::addInputs(node, "src", src);
      jit::tracer::addInputs(node, "self", self);
      graph->insertNode(node);
      jit::tracer::ensureUniqueIfOutOfPlaced("copy_ (possibly due to an assignment)", self);
      output = node->output();
    } else {
      output = graph->insert(
          jit::aten::copy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
    }
  }
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.scalar_type());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.device();
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->copy_(self_, src_, non_blocking);
  }
  increment_version(self);
  rebase_history(as_variable_ref( self ), std::move(grad_fn));
  if(torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  return self;
}

Tensor & VariableType::resize_(Tensor & self, IntArrayRef size) const {
  auto& self_ = unpack(self, "self", 0);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::ArgumentStash::popIntArrayRef("size");
    jit::tracer::warn("resize_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->resize_(self_, size);
  }
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) const {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::warn("resize_as_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->resize_as_(self_, the_template_);
  }
  return self;
}

Tensor VariableType::detach(const Tensor & self) const {
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}));

  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);

  }
  // <NON_GENERATED_CODE>
  auto result = as_variable_ref(const_cast<Tensor&>(self)).detach(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
  // </NON_GENERATED_CODE>
  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, result);
  }
  return std::move(result);
}

Tensor & VariableType::detach_(Tensor & self) const {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));

  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_", self);
  }
  // <NON_GENERATED_CODE>
  as_variable_ref(self).detach_();
  // </NON_GENERATED_CODE>
  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, self);
  }
  return self;
}

Tensor VariableType::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) const {
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims_and_tensors", std::vector<c10::IValue>({indices, values}), Function::peek_at_next_sequence_nr());

  // SparseTensorImpl's `set_indices_and_values_unsafe(indices, values)` function expects
  // `indices` and `values` to not require grad. The `unpack()` function calls in the
  // auto-generated version of this function doesn't always return `indices_` and `values_`
  // with requires_grad=false, and we need another way to create non-requires-grad `indices_`
  // and `values_` tensors that contains the same data as `indices` and `values`, which is
  // achieved by shallow-copying `indices` and `values` and wrapping them in new Tensors here.
  auto indices_ = Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach());
  auto values_ = Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach());
  auto options_ = TensorOptions(options).is_variable(false);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward> grad_fn;
  if (compute_requires_grad( values )) {
    grad_fn = std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward>(new SparseCooTensorWithDimsAndTensorsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( values ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->values_sizes = values.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices_, values_, options_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

}} // namespace torch::autograd
