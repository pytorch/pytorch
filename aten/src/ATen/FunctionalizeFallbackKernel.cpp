#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>
#include <c10/util/irange.h>

namespace {
  void functionalizeFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    TORCH_INTERNAL_ASSERT(!schema.hasAnyAliasInfo(), "mutating and aliasing ops should all have codegen'd kernels");
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);

    for (uint64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        auto t = ivalue.toTensor();
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

    for (const auto idx : c10::irange(num_returns)) {
      const auto& ivalue = returns[idx];
      if (ivalue.isTensor()) {
        auto t = ivalue.toTensor();
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

TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}
