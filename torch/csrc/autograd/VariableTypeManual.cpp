#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/autograd/utils/error_messages.h>

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

namespace {
std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  std::vector<DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (int64_t s = 0; s < static_cast<int64_t>(ScalarType::NumOptions); s++) {
      auto& type = getDeprecatedTypeProperties(static_cast<Backend>(p), static_cast<ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}
}

namespace VariableType {

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCPUTypes() {
  return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({ Backend::CUDA, Backend::SparseCUDA });
}

const Variable & checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

Variable & checked_cast_variable(Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

const Tensor & unpack(const Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor & unpack(Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor unpack_opt(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    }
    if (!t.is_variable()) {
      AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " at position #", i, " "
                    "for iterable argument #", pos, " '", name, "'");
    }
    ret[i] = static_cast<const Variable&>(t);
  }
  return ret;
}

void backward(
    const Tensor& self,
    const Tensor& gradient,
    bool keep_graph,
    bool create_graph) {
  as_variable_ref(self).backward(gradient, keep_graph, create_graph);
}

void set_data(const Tensor & self, const Tensor & new_data) {
  as_variable_ref(self).set_data(new_data);
}

Tensor data(const Tensor & self) {
  return as_variable_ref(self).variable_data();
}

bool is_leaf(const Tensor & self) {
  return as_variable_ref(self).is_leaf();
}

int64_t output_nr(const Tensor & self) {
  return as_variable_ref(self).output_nr();
}

int64_t _version(const Tensor & self) {
  return as_variable_ref(self).current_version();
}

Tensor& requires_grad_(Tensor& self, bool _requires_grad) {
  if (!self.is_leaf() && !_requires_grad) {
    throw std::runtime_error(
      autograd::utils::requires_grad_leaf_error(_requires_grad)
    );
  }
  return self.set_requires_grad(_requires_grad);
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
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
  // currently, isFloatingType will return false for (floating) complex types,
  // so this might have to be amended when they should be differentiable
  requires_grad &= isFloatingType(self.scalar_type());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->src_options = src.options();
    grad_fn->src_device = src.device();
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.copy_(src_, non_blocking);
  }
  increment_version(self);
  rebase_history(as_variable_ref( self ), std::move(grad_fn));
  if(torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  return self;
}

Tensor & resize_(Tensor & self, IntArrayRef size) {
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
    self_.resize_(size);
  }
  return self;
}

Tensor & resize_as_(Tensor & self, const Tensor & the_template) {
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
    at::resize_as_(self_, the_template_);
  }
  return self;
}

Tensor detach(const Tensor & self) {
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

Tensor & detach_(Tensor & self) {
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

}  // namespace VariableType

}} // namespace torch::autograd
