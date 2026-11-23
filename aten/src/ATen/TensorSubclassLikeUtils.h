#pragma once
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/equal.h>
#endif

namespace at {

// Note [Tensor-subclass-like Tensors]
// Tensor-subclass-like is defined as:
// - a Tensor subclass (via __torch_dispatch__ in Python or extending
//   TensorImpl in C++)
// - anything else that shares the same perils as Tensor subclasses.
//   For example, many Tensor subclasses do not have storage and meta Tensors
//   do not have storage either, so meta Tensors belong here.
//
// We should ensure that PyTorch internals supports Tensor-subclass-like
// objects. In particular, Tensor-subclass-like objects struggle with two
// classes of operations that are problematic for Tensor subclasses:
// 1. Because some Tensor subclasses do not have storage, .item() or
//    .data_ptr() calls are not good.
// 2. Certain in-place operations can eliminate the typing of the Tensor
//    subclass. For example:
//    >>> torch.zeros(input.sizes(), grad.options()).diag().copy_(input)
//    If input is a Tensor subclass, then the above ends up either erroring out
//    or returning a regular non-Tensor-subclass Tensor!

constexpr auto kFunctorchWrappedTensors = DispatchKeySet(
    {DispatchKey::FuncTorchGradWrapper,
     DispatchKey::FuncTorchBatched,
     DispatchKey::Functionalize});

constexpr auto kTensorSubclassLike =
    kFunctorchWrappedTensors |
    DispatchKeySet(
        {// WARNING: DO NOT put combined backend component + functionality keys
         // here, you will incorrectly always match on the functionality key
         // no matter the backend component
         DispatchKey::Batched,
         DispatchKey::Sparse,
         DispatchKey::SparseCsr,
         DispatchKey::Python,
         // isTensorSubclassLike decides whether a kernel can take a fast path
         // for plain dense tensors that skips the dispatcher. That is unsafe
         // for FakeTensors, which Python FakeTensor gets via the Python key.
         // C++ FakeTensor only has the Fake key but is still a FakeTensor (its
         // ops must reach the Fake fallback), so Fake is listed to match.
         DispatchKey::Fake}) |
    DispatchKeySet(BackendComponent::MetaBit);

// Like a Python mode (dispatch_mode_enabled()), an active C++ FakeTensorMode
// intercepts ops on every tensor, including non-fake ones, so all tensors are
// subclass-like while it is active. It is not on the Python mode stack; it is
// active when Fake is in the TLS include set and not excluded (the Fake
// fallback excludes Fake while it runs meta/real kernels).
inline bool anyModeEnabled() {
  return c10::impl::dispatch_mode_enabled() ||
      (c10::impl::tls_is_dispatch_key_included(DispatchKey::Fake) &&
       !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Fake));
}

inline bool isTensorSubclassLike(const Tensor& tensor) {
  if (anyModeEnabled())
    return true;
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  return !(key_set & kTensorSubclassLike).empty();
}

inline bool areAnyTensorSubclassLike(TensorList tensors) {
  if (anyModeEnabled())
    return true;
  return std::any_of(tensors.begin(), tensors.end(), isTensorSubclassLike);
}

inline bool areAnyOptionalTensorSubclassLike(
    const c10::List<std::optional<Tensor>>& tensors) {
  if (anyModeEnabled())
    return true;
  return std::any_of(
      tensors.begin(),
      tensors.end(),
      [](const std::optional<Tensor>& opt_tensor) {
        return (
            opt_tensor.has_value() && isTensorSubclassLike(opt_tensor.value()));
      });
}

// Helper function to deal testing truthfulness of a scalar tensor
// in a Composite Compliant manner.
// NOTE: This function expects a scalar tensor of boolean dtype.
// Eg.
// Non-Composite Compliant Pattern : (t == 0).all().item<bool>()
// Composite Compliant Pattern : is_salar_tensor_true((t == 0).all())
inline bool is_scalar_tensor_true(const Tensor& t) {
  TORCH_INTERNAL_ASSERT(t.dim() == 0)
  TORCH_INTERNAL_ASSERT(t.scalar_type() == kBool)
  return at::equal(t, t.new_ones({}, t.options()));
}

} // namespace at
