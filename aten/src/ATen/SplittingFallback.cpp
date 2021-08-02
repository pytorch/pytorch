#include <torch/library.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ATen.h>
#include <c10/util/irange.h>

namespace at {

void splittable_rng_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  const auto& schema = op.schema();

  op.redispatchBoxed(dispatch_keys.remove(DispatchKey::SplittableRNGKey), stack);

  const auto num_returns = schema.returns().size();
  const auto stack_start = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    auto& ivalue = (*stack)[stack_start + i];
    if (!ivalue.isTensor()) {
      continue;
    }
    auto tensor = std::move(ivalue).toTensor();
    tensor._set_rng_key(true);
    (*stack)[stack_start + i] = std::move(tensor);
  }
}

TORCH_LIBRARY_IMPL(_, SplittableRNGKey, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&splittable_rng_fallback>());
}

} // namespace at
