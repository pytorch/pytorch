#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/TracerMode.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/irange.h>
#include <torch/library.h>

using namespace at;
using namespace torch::autograd::generated;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;

namespace torch { namespace autograd { namespace VariableType {

std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  std::vector<DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for(const auto s : c10::irange(static_cast<int64_t>(ScalarType::NumOptions))) {
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

// Taken from codegened version
Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor & self, int64_t level) {
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Identity> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Identity>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }

  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::_fw_primal(ks & c10::after_autograd_keyset, self_, level);
  })();

  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (generated::details::isFwGradDefined(self)) {
    // Modified from original codegen
    // We explicitly want to ignore the forward grad at the given level
    TORCH_CHECK(level == 0, "Invalid level given to _fw_primal");
    // End modified from original codegen
  }
  return result;
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor & copy_(c10::DispatchKeySet ks, Tensor & self, const Tensor & src, bool non_blocking) {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isDifferentiableType(self.scalar_type());
  check_inplace(self, requires_grad);
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->src_options = src.options();
    grad_fn->src_device = src.device();
  }
  {
    at::AutoDispatchBelowAutograd mode;
    at::redispatch::copy_(ks & c10::after_autograd_keyset, self_, src_, non_blocking);
  }
  rebase_history(self , std::move(grad_fn));

  if (isDifferentiableType(self.scalar_type()) &&
      (generated::details::isFwGradDefined(self) || generated::details::isFwGradDefined(src))) {
    auto self_fw_grad = generated::details::toNonOptFwGrad(self);
    auto src_fw_grad = generated::details::toNonOptFwGrad(src);
    Tensor new_fw_grad;
    if (self_fw_grad.defined()) {
      if (src_fw_grad.defined()) {
        new_fw_grad = self_fw_grad.copy_(src_fw_grad);
      } else {
        new_fw_grad = self_fw_grad.fill_(0);
      }
    } else {
      new_fw_grad = src_fw_grad;
    }
    self._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
  }

  return self;
}

const Tensor& resize_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto& self_ = unpack(self, "self", 0);
  if (self.requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoDispatchBelowAutograd mode;
    at::redispatch::resize_(ks & c10::after_autograd_keyset, self_, size, optional_memory_format);
  }

  if (self._fw_grad(/* level */ 0).defined()) {
    AT_ERROR("cannot resize variables that has a forward grad");
  }

  return self;
}

const Tensor& resize_as_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (self.requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoDispatchBelowAutograd mode;
    at::redispatch::resize_as_(ks & c10::after_autograd_keyset, self_, the_template_, optional_memory_format);
  }

  // Handle fw grad
  if (self._fw_grad(/* level */ 0).defined()) {
    AT_ERROR("cannot resize variables that has a forward grad");
  }
  return self;
}

Tensor detach(c10::DispatchKeySet ks, const Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}));
  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::detach(ks & c10::after_autograd_keyset, self_);
  })();
  namedinference::propagate_names(result, self);

  // Detach the forward grads by not setting anything on the result

  return result;
}

Tensor & detach_(c10::DispatchKeySet ks, Tensor & self) {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));
  if (self.is_view()) {
    // NB: is_view() ==> get_autograd_meta()
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(self));
    // See NOTE [ View + Inplace detection ]
    if (diff_view_meta->get_creation_meta() == CreationMeta::MULTI_OUTPUT_SAFE) {
        TORCH_WARN("This view is an output of a function that "
                   "returns multiple views. Detaching such views inplace "
                   "is being deprecated and will be forbidden "
                   "starting from version 1.8. Consider using detach() instead "
                   "of detach_(). Alternatively, create this view with an "
                   "`unsafe_` version of the function that produced it.");
    } else {
      AT_ERROR("Can't detach views in-place. Use detach() instead. "
               "If you are using DistributedDataParallel (DDP) for training, "
               "and gradient_as_bucket_view is set as True, gradients are "
               "views of DDP buckets, and hence detach_() cannot be called "
               "on these gradients. To fix this error, please refer to the "
               "Optimizer.zero_grad() function in torch/optim/optimizer.py "
               "as the solution.");
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
  autograd_meta->fw_grad_.reset();

  return self;
}

// Ops in the following registration list are registered as
//   (1) CompositeImplicitAutograd kernels
//   (2) Autograd kernels
//   (3) CompositeExplicitAutograd kernels and additionally Autograd kernels
// The reason for (3) is that ops that also use dispatch (e.g. register CPU/CUDA/QuantizedCPU
// kernels) will skip picking up CompositeImplicitAutograd kernels for Autograd, so we register them to both
// CompositeExplicitAutograd and Autograd instead. See
// https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword
// for more details.
// Invariant:
// - Ops registered to CompositeImplicitAutograd or CompositeExplicitAutograd below must match `MANUAL_BACKEND` set in tools/autograd/gen_variable_type.py.
//   and they have manual_kernel_registration=True in native_functions.yaml.
// - Ops registered to DispatchKey::Autograd below must be included in `MANUAL_AUTOGRAD` in tools/autograd/gen_variable_type.py

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("resize_", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::resize_)));
  m.impl("resize_as_", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::resize_as_)));
  m.impl("detach", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach)));
  m.impl("detach_", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach_)));
  m.impl("copy_", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::copy_)));
  m.impl("_fw_primal", torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::_fw_primal)));
}

}  // namespace
}} // namespace autograd::VariableType

namespace ADInplaceOrView {
  #define CREATION_META_DEFINITION InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE)

  Tensor & copy_(c10::DispatchKeySet ks, Tensor & self, const Tensor & src, bool non_blocking) {
    {
      at::AutoDispatchBelowADInplaceOrView guard;
      at::redispatch::copy_(ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking);
    }
    torch::autograd::increment_version(self);
    return self;
  }

  Tensor detach(c10::DispatchKeySet ks, const Tensor & self) {
    auto out = ([&]() {
      at::AutoDispatchBelowADInplaceOrView guard;
      // Make an empty shallow copy, the as_view call below will fill in the proper fields
      return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/false));
    })();
    std::function<at::Tensor(const at::Tensor&)> func=nullptr;
    auto result = as_view(/* base */ self, /* output */ out, /* is_bw_differentiable */ false,
                          /* is_fw_differentiable */ false, /* view_func */ func,
                          /* creation_meta */ CreationMeta::DEFAULT,
                          /*allow_tensor_metadata_change=*/false);

    return result;
  }

  Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor & self, int64_t level) {
    auto tmp = ([&]() {
      at::AutoDispatchBelowADInplaceOrView guard;
      // Make an empty shallow copy, the as_view call below will fill in the proper fields
      return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/false));
    })();
    std::function<at::Tensor(const at::Tensor&)> func=nullptr;
    if (!self.unsafeGetTensorImpl()->support_as_strided()) {
      auto size_vec = self.sizes().vec();
      func = [=](const at::Tensor& input_base) {
        return input_base.view(size_vec);
      };
    }
    auto result = as_view(/* base */ self, /* output */ tmp, /* is_bw_differentiable */ true,
                          /* is_fw_differentiable */ false, /* view_func */ func, /* creation_meta */ CREATION_META_DEFINITION);

    return result;
  }

  namespace {
    TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
      m.impl("copy_", torch::dispatch(DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::copy_)));
      m.impl("detach", torch::dispatch(DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::detach)));
      m.impl("_fw_primal", torch::dispatch(DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::_fw_primal)));
    }
  } // namespace
} // namespace ADInplaceOrView
} // namespace torch
