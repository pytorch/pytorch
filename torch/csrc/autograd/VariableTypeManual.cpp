#include <ATen/RedispatchFunctions.h>
#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/utils/memory.h>
#include <torch/library.h>

#include <utility>

using namespace at;
using namespace torch::autograd::generated;
using torch::autograd::as_view;
using torch::autograd::CreationMeta;

namespace torch {
namespace autograd {
namespace VariableType {

std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(
    at::ArrayRef<at::Backend> backends) {
  std::vector<DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (const auto s :
         c10::irange(static_cast<int64_t>(ScalarType::NumOptions))) {
      auto& type = getDeprecatedTypeProperties(
          static_cast<Backend>(p), static_cast<ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCPUTypes() {
  return allTypesForBackends({Backend::CPU, Backend::SparseCPU});
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({Backend::CUDA, Backend::SparseCUDA});
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allXPUTypes() {
  return allTypesForBackends({Backend::XPU, Backend::SparseXPU});
}

namespace {
const Variable& checked_cast_variable(
    const Tensor& t,
    const char* name,
    int pos) {
  if (!t.defined()) {
    AT_ERROR(
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}

Variable& checked_cast_variable(Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    AT_ERROR(
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}
} // namespace

const Tensor& unpack(const Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor& unpack(Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor unpack_opt(const Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> unpack(
    at::ITensorListRef tl,
    const char* name,
    int pos) {
  std::vector<at::Tensor> ret;
  ret.reserve(tl.size());
  for (const auto& t : tl) {
    ret.push_back(t.defined() ? static_cast<const Variable&>(t) : Variable{});
  }
  return ret;
}

namespace {

// Taken from codegened version
Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor& self, int64_t level) {
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Identity> grad_fn;
  if (compute_requires_grad(self)) {
    grad_fn = std::make_shared<Identity>();
    grad_fn->set_next_edges(collect_next_edges(self));
  }

  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::_fw_primal(
        ks & c10::after_autograd_keyset, self_, level);
  })();

  if (grad_fn) {
    set_history(flatten_tensor_args(result), grad_fn);
  }
  if (isFwGradDefined(self)) {
    // Modified from original codegen
    // We explicitly want to ignore the forward grad at the given level
    TORCH_CHECK(level == 0, "Invalid level given to _fw_primal");
    // End modified from original codegen
  }
  return result;
}

// NB: We need a manual variable type kernel so that set_fw_grad properly
// detects that _make_dual is not a forward-differentiable view
//
// This function can be used to create a dual Tensor that holds a tangent to
// compute forward mode gradients. Note that the dual Tensor's primal is a view
// of the given primal and the given tangent is used as-is. This function is
// backward differentiable.
Tensor _make_dual(
    c10::DispatchKeySet ks,
    const Tensor& primal,
    const Tensor& tangent,
    int64_t level) {
  TORCH_CHECK(
      !primal._fw_grad(level).defined(),
      "Making a dual Tensor based on a Tensor that "
      "already has a forward gradient at the same level ",
      level,
      " is not supported.");
  auto& primal_ = unpack(primal, "primal", 0);
  auto& tangent_ = unpack(tangent, "tangent", 0);
  std::shared_ptr<ViewBackward0> grad_fn;
  if (compute_requires_grad(primal_)) {
    grad_fn = std::make_shared<ViewBackward0>();
    grad_fn->self_sym_sizes = primal_.sym_sizes().vec();
    grad_fn->set_next_edges(collect_next_edges(primal_));
  }

  auto result = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::_make_dual(
        ks & c10::after_autograd_keyset, primal_, tangent_, level);
  })();

  if (grad_fn) {
    set_history(flatten_tensor_args(result), grad_fn);
  }

  TORCH_CHECK(level == 0, "Invalid level given to _make_dual");
  result._set_fw_grad(tangent_, level, /* is_inplace_op */ false);
  return result;
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor& copy_(
    c10::DispatchKeySet ks,
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
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
  }
  {
    at::AutoDispatchBelowAutograd mode;
    at::redispatch::copy_(
        ks & c10::after_autograd_keyset, self_, src_, non_blocking);
  }
  rebase_history(self, std::move(grad_fn));

  if (isDifferentiableType(self.scalar_type()) &&
      (isFwGradDefined(self) || isFwGradDefined(src))) {
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
      if (!self.is_same_size(src_fw_grad)) {
        new_fw_grad = src_fw_grad.broadcast_to(self.sizes());
      } else {
        new_fw_grad = src_fw_grad.clone();
      }
    }
    self._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
  }

  return self;
}

const Tensor& resize_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    SymIntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto& self_ = unpack(self, "self", 0);
  if (self.requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  {
    at::AutoDispatchBelowAutograd mode;
    at::redispatch::resize__symint(
        ks & c10::after_autograd_keyset, self_, size, optional_memory_format);
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
    at::redispatch::resize_as_(
        ks & c10::after_autograd_keyset,
        self_,
        the_template_,
        optional_memory_format);
  }

  // Handle fw grad
  if (self._fw_grad(/* level */ 0).defined()) {
    AT_ERROR("cannot resize variables that has a forward grad");
  }

  return self;
}

Tensor detach(c10::DispatchKeySet ks, const Tensor& self) {
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

Tensor& detach_(c10::DispatchKeySet ks, Tensor& self) {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));
  if (self.is_view()) {
    // See NOTE [ View + Inplace detection ]
    AT_ERROR(
        "Can't detach views in-place. Use detach() instead. "
        "If you are using DistributedDataParallel (DDP) for training, "
        "and gradient_as_bucket_view is set as True, gradients are "
        "views of DDP buckets, and hence detach_() cannot be called "
        "on these gradients. To fix this error, please refer to the "
        "Optimizer.zero_grad() function in torch/optim/optimizer.py "
        "as the solution.");
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
// The reason for (3) is that ops that also use dispatch (e.g. register
// CPU/CUDA/QuantizedCPU kernels) will skip picking up CompositeImplicitAutograd
// kernels for Autograd, so we register them to both CompositeExplicitAutograd
// and Autograd instead. See
// https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword
// for more details.
// Invariant:
// - Ops registered to CompositeImplicitAutograd or CompositeExplicitAutograd
// below must match `MANUAL_BACKEND` set in tools/autograd/gen_variable_type.py.
//   and they have manual_kernel_registration=True in native_functions.yaml.
// - Ops registered to DispatchKey::Autograd below must be included in
// `MANUAL_AUTOGRAD` in tools/autograd/gen_variable_type.py

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl(
      "resize_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::resize_)));
  m.impl(
      "resize_as_",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::resize_as_)));
  m.impl(
      "detach",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach)));
  m.impl(
      "detach_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::detach_)));
  m.impl(
      "copy_",
      torch::dispatch(DispatchKey::Autograd, TORCH_FN(VariableType::copy_)));
  m.impl(
      "_fw_primal",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::_fw_primal)));
  m.impl(
      "_make_dual",
      torch::dispatch(
          DispatchKey::Autograd, TORCH_FN(VariableType::_make_dual)));
}

} // namespace
} // namespace VariableType
} // namespace autograd

namespace ADInplaceOrView {
#define CREATION_META_DEFINITION                            \
  InferenceMode::is_enabled()                               \
      ? CreationMeta::INFERENCE_MODE                        \
      : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT \
                                    : CreationMeta::NO_GRAD_MODE)

Tensor& copy_(
    c10::DispatchKeySet ks,
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copy_(
        ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking);
  }
  torch::autograd::increment_version(self);
  return self;
}

const Tensor& resize_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    SymIntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  // Hold sizes to verify if we actually resize `self`.
  // Explicitly copy data, since resizing can move original data
  // and make references invalid.
  auto org_size = self.sym_sizes().vec();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::resize__symint(
        ks & c10::after_ADInplaceOrView_keyset,
        self,
        size,
        optional_memory_format);
  }
  // If `self` was resized, increment the version.
  if (org_size != size) {
    torch::autograd::increment_version(self);
  }
  return self;
}

const Tensor& resize_as_(
    c10::DispatchKeySet ks,
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  // Hold sizes to verify if we actually resize `self`.
  // Explicitly copy data, since resizing can move original data
  // and make references invalid.
  auto org_size = self.sym_sizes().vec();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::resize_as_(
        ks & c10::after_ADInplaceOrView_keyset,
        self,
        the_template,
        optional_memory_format);
  }

  // If `self` was resized, increment the version.
  if (org_size != the_template.sym_sizes()) {
    torch::autograd::increment_version(self);
  }
  return self;
}

Tensor detach(c10::DispatchKeySet ks, const Tensor& self) {
  auto out = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::detach::redispatch(
        ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  // NB: we can't make detach() a normal view operator because the codegen
  // generates allow_tensor_metadata_change = True for them. In the future we
  // should have an option for this in the codegen.
  std::function<at::Tensor(const at::Tensor&)> func = nullptr;
  auto result = as_view(
      /* base */ self,
      /* output */ out,
      /* is_bw_differentiable */ false,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* creation_meta */ CreationMeta::DEFAULT,
      /*allow_tensor_metadata_change=*/false);

  return result;
}

Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor& self, int64_t level) {
  auto tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::alias(self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func = nullptr;
  if (!self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = self.sizes().vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.view(size_vec);
    };
  }
  auto result = as_view(
      /* base */ self,
      /* output */ tmp,
      /* is_bw_differentiable */ true,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* creation_meta */ CREATION_META_DEFINITION);

  return result;
}

// NB: This does not redispatch any further
Tensor _make_dual(
    c10::DispatchKeySet ks,
    const Tensor& primal,
    const Tensor& tangent,
    int64_t level) {
  auto tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::alias(primal);
  })();
  std::function<at::Tensor(const at::Tensor&)> func = nullptr;
  if (!primal.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = primal.sizes().vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.view(size_vec);
    };
  }
  auto result = as_view(
      /* base */ primal,
      /* output */ tmp,
      /* is_bw_differentiable */ true,
      /* is_fw_differentiable */ false,
      /* view_func */ std::move(func),
      /* creation_meta */ CREATION_META_DEFINITION);

  return result;
}

namespace {
TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl(
      "copy_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::copy_)));
  m.impl(
      "detach",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::detach)));
  m.impl(
      "resize_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::resize_)));
  m.impl(
      "resize_as_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::resize_as_)));
  m.impl(
      "_fw_primal",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::_fw_primal)));
  m.impl(
      "_make_dual",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::_make_dual)));
}
} // namespace
} // namespace ADInplaceOrView
} // namespace torch
