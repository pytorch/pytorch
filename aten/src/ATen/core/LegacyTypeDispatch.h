#pragma once

// The legacy mechanism for dispatching operators in ATen is a Type
// object, which is essentially a giant virtual dispatch table
// for every operation we support dynamically dispatching over.
//
// This has been deprecated in favor of ATenDispatch, and in the future,
// c10 dispatcher.
// TODO: Clean up what remains here

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {

// A RAII, thread local (!) guard that will disable dispatch to variable
// handler.
//
// NOTE [ Treating Variables as non-Variables in type dispatch ]
//
// What exactly does AutoDispatchBelowAutograd do?  The short answer is, it causes
// dispatches on ATen functions to go to the non-variable implementation,
// bypassing autograd handling (and also profiling and tracing).
//
// To understand why this guard exists, it's helpful to understand the history
// behind how Variable was implemented.  Previously, Variables were implemented
// as a wrapper on Tensors; so the act of processing a Variable involved
// unwrapping the underlying Tensor, and then calling the underlying base
// operation on /that/ operation
//
// However, after the Variable/Tensor merge, there is no concept of unwrapping
// a tensor anymore.  If you just call the operation on the same variable
// again inside your VariableType handler, you'll dispatch back to
// VariableType, which is not what we want.
//
// The solution to the above problem is to add `at::AutoDispatchBelowAutograd`, which
// when enabled will cause `legacyTensorType()` and `getType()` to always return
// non-Variable type, even if the tensor being called on is a variable.

/* Note [AutoDispatchBelowAutograd]
 * AutoDispatchBelowAutograd is **INTERNAL ONLY** that it should be used
 * for kernel implementations and customized C++ kernels.
 * If you are looking for a guard to run workload in inference mode, please use
 * c10::InferenceMode RAII which is user facing API.
 * In the past AutoDispatchBelowAutograd(or its old version AutoNonVariableTypeMode)
 * was used in the user code for inference-only workload, this was under risk of
 * producing wrong results silently in some edge cases. For example:
 * ```
 *  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
 *  torch::Tensor out = s * s;
 *  {
 *    at::AutoDispatchBelowAutograd guard;
 *    s.add_(1);  // Skips version bump on `s`.
 *  }
 *  // WRONG GRADIENT! s.grad() are now computed using `s` value after the
 *  // inplace update.
 *  out.backward(torch::ones_like(out));
 * ```
 * Users should use `c10::InferenceMode` here so that it'll properly throw an
 * error saying "one of the variables needed for gradient computation has be modified."
 */
struct TORCH_API AutoDispatchBelowAutograd {
  AutoDispatchBelowAutograd() :
    autograd_guard_(c10::autograd_dispatch_keyset) {
  }

  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

// TODO: AutoNonVariableTypeMode should be removed in release 1.10.
struct TORCH_API AutoNonVariableTypeMode {
  AutoNonVariableTypeMode(bool enabled = true) :
    autograd_guard_(c10::autograd_dispatch_keyset) {
    TORCH_WARN_ONCE("AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. "
        "For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, "
        "If you are looking for a user facing API to enable running your inference-only "
        "workload, please use c10::InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code "
        "is under risk of producing silent wrong result in some edge cases. "
        "See Note [AutoDispatchBelowAutograd] for more details.");
    TORCH_INTERNAL_ASSERT(enabled);
  }

  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

/* Note [AutoDispatchBelowADInplaceOrView]
 * AutoDispatchBelowADInplaceOrView is equivalent to AutoNonVariableTypeMode
 * before we split inplace & view ops out of VariableType kernel.
 * Note this guard is used in VariableType kernels for functional ops
 * as well as ADInplaceOrView kernels for inplace/view ops to enforce the
 * Invariant:
 *   Once you are in VariableType/ADInplaceOrView kernel for an op,
 *   you never go back to a kernel on same dispatch key until
 *   you finish the current op.
 */
struct TORCH_API AutoDispatchBelowADInplaceOrView {
  AutoDispatchBelowADInplaceOrView() :
    dispatch_key_guard_(c10::autograd_dispatch_keyset_with_ADInplaceOrView) {
  }
  // disable Autograd & ADInplaceOrView dispatch keys
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};
} // namespace at
