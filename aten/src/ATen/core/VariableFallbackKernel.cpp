#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorImpl.h>
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

	// Note [Wrapping return values in the functionalization pass]
	// We need to read from TLS in order to distinguish between the functorch case and the XLA case.
	// In functorch's functionalization pass, we (need to) add the key to the TLS include set.
	// We need to do this in order to ensure that free variables created from factory functions
	// inside of the program are properly wrapped.
	// However, we DO NOT want to wrap outputs unconditionally.
	// For example, when calling xla_tensor.to('cpu'), the output should not be wrapped.
	// TLS is the only way to distinguish between these two cases.
	bool is_modal_pass = c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize);

    for (int64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        at::Tensor t = ivalue.toTensor();

		at::functionalization::impl::maybe_sync(t);
		auto maybe_unwrapped = at::functionalization::impl::maybeUnwrapFunctional(t);
		auto materialized_ivalue = c10::IValue(maybe_unwrapped);
		(*stack)[arguments_begin + idx] = std::move(materialized_ivalue);

      } else if (ivalue.isTensorList()) {
        auto tensors = ivalue.toTensorList();
		at::functionalization::impl::maybe_sync(tensors);
		auto maybe_unwrapped = at::functionalization::impl::maybeUnwrapFunctional(tensors);
		auto materialized_ivalue = c10::IValue(maybe_unwrapped);
		(*stack)[arguments_begin + idx] = std::move(materialized_ivalue);
      }
    }
    {
      at::AutoDispatchBelowFunctionalize guard;
      // redispatchBoxed with specified dispatchKeySet cannot prevent composite kernels
      // called inside from going back up dispatcher. We still need the RAII guard here.
      op.redispatchBoxed(dispatchKeySet & c10::after_func_keyset, stack);
    }

    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    auto returns = torch::jit::last(stack, num_returns);
    int ctr = 0;
    for (int64_t idx = 0; idx < num_returns; ++idx) {
      const auto& ivalue = returns[idx];
      if (ivalue.isTensor()) {
        at::Tensor t = ivalue.toTensor();
        if (is_modal_pass) {
          auto wrapped_t = at::functionalization::impl::makeFunctional(t);
          auto wrapped_ivalue = c10::IValue(wrapped_t);
          (*stack)[returns_begin + idx] = std::move(wrapped_ivalue);
        }
      } else if (ivalue.isTensorList()) {
        auto tensors = ivalue.toTensorList();
        if (is_modal_pass) {
          auto tensors = ivalue.toTensorList();
          auto wrapped_t = at::functionalization::impl::makeFunctional(tensors);
          auto wrapped_ivalue = c10::IValue(wrapped_t);
          (*stack)[returns_begin + idx] = std::move(wrapped_ivalue);
        }
      }
    }
  }
}

TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}

// see Note [ADInplaceOrView key]
TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
      m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  // We need this for the functionalization pass: replace_ shouldn't enter the boxed fallback
  m.impl("replace_", torch::CppFunction::makeFallthrough());
}

}
