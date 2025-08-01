#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

// TODO: Refer to this https://github.com/pytorch/pytorch/issues/29548 to
// consider whether this file should be deleted and replaced with the mechanism
// described above

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

using c10::Stack;

#ifdef C10_MOBILE

// NOTE [mobile/edge builds and the autograd fallback]
// To save on binary size, some of the mobile configs don't include the
// autograd kernels for built-in operators (VariableTypeEverything.cpp).
// For the mobile build:
// - we don't care about having a nice autograd fallback that warns if
// an operator has incorrect autograd support. If you're running
// a custom operator on mobile then it's already too late for us to warn
// or error on it.
// - for perf reasons, we do not want mobile to go through autograd_fallback
// for all operators (the boxing/unboxing adds overhead).
// As a result, on mobile we set the fallback to the fallthrough.
#define AUTOGRAD_FALLBACK torch::CppFunction::makeFallthrough()

#else

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!
C10_API void autograd_fallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack);

#define AUTOGRAD_FALLBACK \
  torch::CppFunction::makeFromBoxedFunction<&autograd_fallback>()

#endif
