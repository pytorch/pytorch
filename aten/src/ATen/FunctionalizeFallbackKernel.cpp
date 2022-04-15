#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/ATen.h>
#else
#include <ATen/ops/_to_copy.h>
#endif
// needed for the meta tensor calls to get stride info in functionalization

namespace {
  void functionalizeFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    TORCH_INTERNAL_ASSERT(!schema.hasAnyAliasInfo(), "mutating and aliasing ops should all have codegen'd kernels");
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);

    auto any_functional_inputs = false;
    auto any_tensor_inputs = false;
    for (uint64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        any_tensor_inputs = true;
        auto t = ivalue.toTensor();
        if (at::functionalization::impl::isFunctionalTensor(t)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(t);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isTensorList()) {
        any_tensor_inputs = true;
        auto tensors = ivalue.toTensorList();
        if (at::functionalization::impl::isFunctionalTensor(tensors)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(tensors);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isOptionalTensorList()) {
        any_tensor_inputs = true;
        auto opt_tensors = ivalue.toOptionalTensorList();
        if (at::functionalization::impl::isFunctionalTensor(opt_tensors)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(opt_tensors);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(opt_tensors));
          (*stack)[arguments_begin + idx] = t_new;
        }
      }
    }
    // we should wrap the output if any inputs were wrapped,
    // OR if we're hitting a factory function (with no tensor inputs)
    auto should_wrap_outputs = !any_tensor_inputs || any_functional_inputs;
    {
      at::AutoDispatchSkipFunctionalize guard;
      op.callBoxed(stack);
    }
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    auto returns = torch::jit::last(stack, num_returns);

    for (const auto idx : c10::irange(num_returns)) {
      const auto& ivalue = returns[idx];
      if (ivalue.isTensor() && should_wrap_outputs) {
        auto t = ivalue.toTensor();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(t));
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isTensorList() && should_wrap_outputs) {
        auto tensors = ivalue.toTensorList();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(tensors));
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isOptionalTensorList() && should_wrap_outputs) {
        auto opt_tensors = ivalue.toOptionalTensorList();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(opt_tensors));
        (*stack)[returns_begin + idx] = t_new;
      }
    }
  }
}


bool device_opted_into_functionalization(c10::optional<c10::Device> d) {
    return d.has_value() && (d->type() == c10::DeviceType::XLA || d->type() == c10::DeviceType::Lazy);
}

at::Tensor _to_copy_functionalize(
        const at::Tensor & self,
        c10::optional<at::ScalarType> dtype,
        c10::optional<at::Layout> layout,
        c10::optional<at::Device> device,
        c10::optional<bool> pin_memory,
        bool non_blocking,
        c10::optional<at::MemoryFormat> memory_format) {
  at::Tensor self_;
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    // sync any pending updates
    at::functionalization::impl::sync(self);
    // pass the unwrapped tensor to the backend
    self_ = at::functionalization::impl::from_functional_tensor(self);
  } else {
    self_ = self;
  }

  at::AutoDispatchSkipFunctionalize guard;
  auto out = at::_to_copy(self_, dtype, layout, device, pin_memory, non_blocking, memory_format);

  // Special case: if the Functionalize key is not in TLS, we assume that we're running
  // on a lazy backend (LTC).
  // In that case, if we're copying to a non-functionalize-enabled device,
  // then the functionalization pass should "end". We need to sync any updates on the input
  // tensor, but we shouldn't wrap the output.
  if (!c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {
    if (!device_opted_into_functionalization(device)) {
      return out;
    }
  }
  return at::functionalization::impl::to_functional_tensor(out);
}



TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}
TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  // Note [Lazy Tensor Functionalization]
  // LazyTensor uses the functionalization pass from core.
  // The first time that we create a lazy tensor
  // (either from a factory function, or from another device using at::to),
  // we need to manually turn the tensor into a "functional tensor" by wrapping it.
  // When we stop performing computation on the lazy device
  // (e.g. when we copy a LazyTensor back onto cpu),
  // we need to "end" the functionalization, syncing any pending updates and unwrapping it.
  m.impl("_to_copy", TORCH_FN(_to_copy_functionalize));
}
