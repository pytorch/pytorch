#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>

/*
 * This file implements a variable fallback kernel for custom operators.
 * Since tensors always have the VariableTensorId set, but custom operators
 * usually don't have a kernel registered for VariableTensorId, the dispatcher
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

static auto registry = Dispatcher::singleton().registerBackendFallbackKernel(
    DispatchKey::VariableTensorId,
    KernelFunction::makeFromBoxedFunction<&variable_fallback_kernel>()
);

}
