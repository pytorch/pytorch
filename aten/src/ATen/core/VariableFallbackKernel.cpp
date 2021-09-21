#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>

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

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

TORCH_LIBRARY_IMPL(_, AutogradOther, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradCUDA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradLazy, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradMLC, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

namespace {
  void functionalizeFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);

    for (int64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        at::Tensor t = ivalue.toTensor();
        at::functionalization::impl::sync(t);
        auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
        (*stack)[arguments_begin + idx] = t_new;
      } else if (ivalue.isTensorList()) {
        auto tensors = ivalue.toTensorList();
        at::functionalization::impl::sync(tensors);
        auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
        (*stack)[arguments_begin + idx] = t_new;
      }
    }
    {
      at::AutoDispatchSkipFunctionalize guard;
      op.redispatchBoxed(dispatchKeySet & c10::after_func_keyset, stack);
    }
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    auto returns = torch::jit::last(stack, num_returns);

    for (int64_t idx = 0; idx < num_returns; ++idx) {
      const auto& ivalue = returns[idx];
      if (ivalue.isTensor()) {
        at::Tensor t = ivalue.toTensor();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(t));
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isTensorList()) {
        auto tensors = ivalue.toTensorList();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(tensors));
        (*stack)[returns_begin + idx] = t_new;
      }
    }
  }
}

at::Tensor& replace_(at::Tensor& self, const at::Tensor& other) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(other));
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  self_impl->replace_(other);
  return self;
}

TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}

// see Note [ADInplaceOrView key]
TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
      m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  // We need this for the functionalization pass: These are all operators used by
  // the functionalization machinery, which themselves shouldn't call into the functionalization pass.
  m.impl("replace_", TORCH_FN(replace_));
}

}
