#pragma once

#include <torch/csrc/jit/runtime/decomposition_registry.h>

// This is the set of helpers in VariableTypeUtils have a dependency on native_functions.yaml
// meaning the file will need to be re-compiled every time an operator is changed or added.
// We cannot simply put these functions in VariableType.h and VariableTypeutils.h, since they are
// included in files like ADInplaceOrViewType_X.cpp which don't /always/ want to be recompiled.

namespace torch {
namespace autograd {
namespace impl {

// Depends on torch/csrc/jit/ir/ir.h -> aten/src/ATen/core/interned_strings.h
template<class Return, class... Args>
Return run_jit_decomposition_with_args(const c10::OperatorHandle& opHandle, c10::DispatchKeySet dispatchKeySet, Args... args) {
  return c10::KernelFunction::makeFromBoxedKernel(c10::BoxedKernel::makeFromFunction<&jit::run_jit_decomposition>()).call<Return, Args...>(opHandle, dispatchKeySet, args...);
}

} // impl
} // autograd
} // torch