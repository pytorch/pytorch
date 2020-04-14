#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>

/*
 * This file implements a variable fallback kernel for custom operators.
 * Since tensors always have the Autograd set, but custom operators
 * usually don't have a kernel registered for Autograd, the dispatcher
 * will call into this fallback kernel instead.
 * Note that this is not a correct autograd implementation. It will just
 * fallthrough to the custom operator implementation.
 * If you want a custom operator to work with autograd, you need to use
 * autograd::Function so that the custom operator implementation knows how to
 * do autograd.
 * Note also that ops from native_functions.yaml register their own variable
 * kernels, so this is never called for them.
 */

// TODO This whole file should be deleted and replaced with the mechanism
//      described in https://github.com/pytorch/pytorch/issues/29548

using c10::OperatorHandle;
using c10::Stack;
using c10::DispatchKey;
using c10::DispatchKeySet;
using c10::Dispatcher;
using c10::KernelFunction;

namespace {

void variable_fallback_kernel(const OperatorHandle& op, Stack* stack) {
    at::AutoNonVariableTypeMode _var_guard(true);
    Dispatcher::singleton().callBoxed(op, stack);
}

static auto registry = Dispatcher::singleton().registerFallback(
    DispatchKey::Autograd,
#ifdef C10_MOBILE
    // As custom mobile build might not include variable kernels, we need
    // leverage variable fallback mechanism as well. The goals are:
    // 1) don't break forward pass for inference-only mobile build;
    // 2) don't break forward/backward pass for mobile build with necessary
    // variable kernels registered;
    //
    // This `fallthrough` kernel is for #1 - because not all kernels support
    // boxed call yet, registering `variable_fallback_kernel` might fail.
    // When an op has variable kernel registered explicitly dispatcher will
    // call it instead of `fallthrough`, so `fallthrough` won't break
    // dispatching to real variable kernels for case #2.
    //
    // The substantial difference between fallback and fallthrough is whether
    // AutoNonVariableTypeMode guard is applied. There are two downstream
    // effects of the guard:
    // a) stop calling variable kernels of other ops called by the current op;
    //    For case #1, there is no difference because no variable kernels are
    //    registered. For case #2, there is no difference as long as ALL used
    //    ops have real variable kernels registered, where the guard will be
    //    set properly in real variable kernels. There is potential issue only
    //    when variable kernels are partially registered for used ops.
    // b) `variable_excluded_from_dispatch()` method returns the state of the
    //    NonVariableTypeMode. As of when this diff is written, the callers of
    //    the method are ALL asserting it returns true; the only exception is
    //    the deprecated `is_variable()` method. So we make the method to always
    //    return true for mobile builds. It shouldn't break case #1/#2 as long
    //    as `is_variable()` is not used.
    //
    // We can remove this `fallthrough` kernel when all kernels support boxed
    // call.
    KernelFunction::makeFallthrough()
#else
    KernelFunction::makeFromBoxedFunction<&variable_fallback_kernel>()
#endif
);

}
