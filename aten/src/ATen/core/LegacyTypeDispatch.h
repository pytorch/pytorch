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
// What exactly does AutoNonVariableType do?  The short answer is, it causes
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
// The solution to the above problem is to add `at::NonVariableTypeMode`, which
// when enabled will cause `legacyTensorType()` and `getType()` to always return
// non-Variable type, even if the tensor being called on is a variable.
//
// TODO: Since `torch::NoGradGuard` serves almost the same purpose in libtorch,
// we should merge these two thread-local guards.  However, NoGradGuard does
// something subtly different: it turns off gradient recording, but DOES NOT
// skip VariableType implementation (as we still might need to profile or
// trace).  To unify the two, we would first have to move profiling and tracing
// out of VariableType.

// TODO: rename this guard and make it internal for kernel implementation only
struct TORCH_API AutoNonVariableTypeMode {
  // NB: The enabled parameter must ALWAYS be black, as Henry Ford used to say.
  // TODO: Eliminate this parameter entirely
  AutoNonVariableTypeMode(bool enabled = true) :
    autograd_guard_(c10::autograd_dispatch_keyset) {

    TORCH_INTERNAL_ASSERT(enabled);
  }

  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

// Note this guard is used in VariableType kernels for functional ops
// as well as InplaceOrView kernels for inplace/view ops to enforce the
// invariant:
//   Once you are in VariableType/InplaceOrView kernel for an op,
//   you never go back to a kernel on same dispatch key until
//   you finish the current op.
struct TORCH_API AutoDispatchBelowInplaceOrView {
  AutoDispatchBelowInplaceOrView() :
    dispatch_key_guard_(c10::autograd_dispatch_keyset_with_InplaceOrView) {
  }
  // disable Autograd & InplaceOrView dispatch keys
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};
} // namespace at
