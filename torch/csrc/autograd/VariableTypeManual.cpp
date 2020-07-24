#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd { namespace VariableType {

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

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCPUTypes() {
  return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({ Backend::CUDA, Backend::SparseCUDA });
}

namespace {
const Variable & checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  return t;
}

Variable & checked_cast_variable(Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  return t;
}
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

c10::optional<Tensor> unpack_opt(const c10::optional<Tensor> & t, const char * name, int pos) {
  return t;
}

std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    }
    ret[i] = static_cast<const Variable&>(t);
  }
  return ret;
}

namespace {

void backward(
    const Tensor& self,
    c10::optional<Tensor> gradient,
    c10::optional<bool> keep_graph,
    bool create_graph) {
  Tensor _gradient = gradient.has_value() ? *gradient : Tensor();
  torch::autograd::backward({self}, {_gradient}, std::move(keep_graph), create_graph);
}

void set_data(const Tensor & self, const Tensor & new_data) {
  // `var.set_data(new_data)` shallow-copies all non-autograd TensorImpl fields
  // from `new_data` to `var`. It requires that `new_data` and `var` have compatible
  // tensor type.
  TORCH_CHECK(
    _has_compatible_shallow_copy_type(self, new_data),
    "Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.");

  // Resets gradient accumulator if metadata is out of date
  AutogradMeta* autograd_meta = impl::get_autograd_meta(self);
  if (autograd_meta) {
    std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
    auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
    if (prior_accumulator) {
      const auto prior_device = prior_accumulator->input_metadata(0).device();
      const auto new_device = new_data.device();

      if (!new_data.options().type_equal(self.options()) || prior_device != new_device) {
        autograd_meta->grad_accumulator_.reset();
      }
    }
  }

  // Version counter is not shared when we replace a `Variable`'s tensor data
  // by calling `set_data(...)`. The original version of the `Variable` is always preserved.
  // See NOTE [ Version Counter Sharing ] for details.
  //
  // `var.set_data(new_data)` always ignores `var`'s `allow_tensor_metadata_change_`, because
  // users need this API as an escape hatch for changing a tensor's metadata regardless of its
  // `allow_tensor_metadata_change_` value, and the users are responsible for ensuring this is
  // the behavior they want.
  self.unsafeGetTensorImpl()->shallow_copy_from(new_data.getIntrusivePtr());
}

Tensor data(const Tensor & self) {
  return self.variable_data();
}

bool is_leaf(const Tensor & self) {
  if (impl::get_autograd_meta(self)) {
    return impl::get_autograd_meta(self)->grad_fn_ == nullptr;
  } else {
    return true;
  }
}

int64_t output_nr(const Tensor & self) {
  if (impl::get_autograd_meta(self)) {
    return impl::get_autograd_meta(self)->output_nr_;
  } else {
    return 0;
  }
}

int64_t _version(const Tensor & self) {
  return self.unsafeGetTensorImpl()->version_counter().current_version();
}

Tensor& requires_grad_(Tensor& self, bool _requires_grad) {
  if (!self.is_leaf() && !_requires_grad) {
    throw std::runtime_error(
      autograd::utils::requires_grad_leaf_error(_requires_grad)
    );
  }
  return self.set_requires_grad(_requires_grad);
}

void retain_grad(const Tensor & self) {
  TORCH_CHECK(self.requires_grad(), "can't retain_grad on Tensor that has requires_grad=False");
  if (self.is_leaf()) {  // no-op for leaves
    return;
  }
  if (impl::get_autograd_meta(self)->retains_grad_) {
    return;
  }
  c10::weak_intrusive_ptr<TensorImpl> weak_self(self.getIntrusivePtr());

  std::function<void(Tensor)> retain_grad_hook([weak_self](const Tensor& grad) {
    if (weak_self.expired()) {
      return;
    } else {
      auto var = weak_self.lock();
      if (!var->grad().defined()) {
        if (grad.is_sparse()) {
          var->mutable_grad() = grad.clone();
        } else {
          var->mutable_grad() = grad.clone(at::MemoryFormat::Contiguous);
        }
      } else {
        var->mutable_grad() = var->grad() + grad;
      }
    }
  });

  self.register_hook(retain_grad_hook);
  impl::get_autograd_meta(self)->retains_grad_ = true;
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
  jit::Value* output = nullptr;
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isDifferentiableType(self.scalar_type());
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
  rebase_history(self , std::move(grad_fn));
  return self;
}

Tensor& resize_(
    Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto& self_ = unpack(self, "self", 0);
  if (self.requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.resize_(size, std::move(optional_memory_format));
  }
  return self;
}

Tensor& resize_as_(
    Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (self.requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::resize_as_(self_, the_template_, std::move(optional_memory_format));
  }
  return self;
}

Tensor detach(const Tensor & self) {
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}));
  auto result = make_variable_non_differentiable_view(self, self, /*allow_tensor_metadata_change=*/false);
  namedinference::propagate_names(result, self);
  return result;
}

Tensor & detach_(Tensor & self) {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));
  if (self.is_view()) {
    // NB: is_view() ==> get_autograd_meta()
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(self));
    // See NOTE [ View + Inplace detection ]
    if (diff_view_meta->creation_meta == CreationMeta::MULTI_OUTPUT_SAFE) {
        TORCH_WARN("This view is an output of a function that "
                   "returns multiple views. Detaching such views inplace "
                   "is being deprecated and will be forbidden "
                   "starting from version 1.8. Consider using detach() instead "
                   "of detach_(). Alternatively, create this view with an "
                   "`unsafe_` version of the function that produced it.");
    } else {
      AT_ERROR("Can't detach views in-place. Use detach() instead");
    }
  }
  // I think the choice here is conservative.  In principle, doing
  // an in-place detach should give us the ability to just clear
  // the autograd meta.  But this function ONLY resets requires_grad,
  // grad_fn and output_nr; there's other metadata like debug name
  // and hooks which aren't cleared.  Is this function supposed to
  // clear those too? I'm not too sure, so I'm leaving it be for now.
  auto autograd_meta = impl::materialize_autograd_meta(self);
  autograd_meta->set_requires_grad(false, self.unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
  return self;
}

// Some ops in the following registration list are registered as catch-all kernels,
// some as catch-all kernels and additionally as backend kernels for Autograd.
// The reason for this is that ops that also use dispatch (e.g. register CPU/CUDA/QuantizedCPU
// kernels) need to get a separate Autograd kernel instead of a catch-all kernel,
// otherwise we won't ever call it for CPU/CUDA/QuantizedCPU tensors, because the backend
// kernel has a higher priority than catch-all kernels.
// Unfortunately, this setup doesn't work in NonVariableTypeMode because that will
// skip past variable kernels. So for ops that we want to use in NonVariableTypeMode
// (and that don't use dispatch), we register them as catch-all kernels instead.
static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .impl_unboxedOnlyKernel<decltype(VariableType::resize_), &VariableType::resize_>(DispatchKey::Autograd))
  .op(torch::RegisterOperators::options()
    .schema("aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .impl_unboxedOnlyKernel<decltype(VariableType::resize_as_), &VariableType::resize_as_>(DispatchKey::Autograd))
  .op(torch::RegisterOperators::options()
    .schema("aten::detach(Tensor(a) self) -> Tensor(a)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .kernel<decltype(VariableType::detach), &VariableType::detach>(DispatchKey::Autograd))
  .op(torch::RegisterOperators::options()
    .schema("aten::detach_(Tensor(a!) self) -> Tensor(a!)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .impl_unboxedOnlyKernel<decltype(VariableType::detach_), &VariableType::detach_>(DispatchKey::Autograd))
  .op(torch::RegisterOperators::options()
    .schema("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .impl_unboxedOnlyKernel<decltype(VariableType::copy_), &VariableType::copy_>(DispatchKey::Autograd))
  .op(torch::RegisterOperators::options()
    .schema("aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    // For backward(), we need the catch-all kernel (see comment above), but we also need the Autograd backend
    // kernel, because when called with a VariableTensorId tensor, it goes through the variable fallback kernel,
    // which calls callBoxed(), which doesn't support optional tensor arguments yet and backward() has an optional
    // tensor argument.
    // TODO Once callBoxed() supports optional tensor arguments, we can enable `use_c10_dispatcher: full` for backward()
    //      and remove the backend Autograd kernel here, only leaving the catch-all kernel.
    .impl_unboxedOnlyKernel<decltype(VariableType::backward), &VariableType::backward>(DispatchKey::Autograd)
    .impl_unboxedOnlyCatchAllKernel<decltype(VariableType::backward), &VariableType::backward>())
  .op(torch::RegisterOperators::options()
    .schema("aten::set_data(Tensor(a!) self, Tensor new_data) -> ()")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::set_data), &VariableType::set_data>())
  .op(torch::RegisterOperators::options()
    .schema("aten::data(Tensor self) -> Tensor")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::data), &VariableType::data>())
  .op(torch::RegisterOperators::options()
    .schema("aten::is_leaf(Tensor self) -> bool")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::is_leaf), &VariableType::is_leaf>())
  .op(torch::RegisterOperators::options()
    .schema("aten::output_nr(Tensor self) -> int")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::output_nr), &VariableType::output_nr>())
  .op(torch::RegisterOperators::options()
    .schema("aten::_version(Tensor self) -> int")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::_version), &VariableType::_version>())
  .op(torch::RegisterOperators::options()
    .schema("aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    // For requires_grad_(), we need the catch-all kernel (see comment above), but we also need the Autograd backend
    // kernel, because when called with a VariableTensorId tensor, it goes through the variable fallback kernel,
    // which calls callBoxed(), which doesn't support mutable tensor arguments yet and requires_grad_() has a mutable
    // tensor argument.
    // TODO Once callBoxed() supports mutable tensor arguments, we can enable `use_c10_dispatcher: full` for requires_grad_()
    //      and remove the backend Autograd kernel here, only leaving the catch-all kernel.
    .impl_unboxedOnlyKernel<decltype(VariableType::requires_grad_), &VariableType::requires_grad_>(DispatchKey::Autograd)
    .impl_unboxedOnlyCatchAllKernel<decltype(VariableType::requires_grad_), &VariableType::requires_grad_>())
  .op(torch::RegisterOperators::options()
    .schema("aten::retain_grad(Tensor(a!) self) -> ()")
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
    .catchAllKernel<decltype(VariableType::retain_grad), &VariableType::retain_grad>())
  ;

}  // namespace
}}} // namespace torch::autograd::VariableType
