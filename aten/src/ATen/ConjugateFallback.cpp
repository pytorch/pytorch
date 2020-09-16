#include <ATen/ConjugateFallback.h>
#include <ATen/native/UnaryOps.h>

namespace at {

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
    // m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
    m.fallback(torch::CppFunction::makeFallthrough());
}

// This function assumes that the tensor input has it's conjugate bit set
Tensor conj_materialize(const Tensor& self) {
  // NOTE: this design is still up in the air- temporarily excluding the conjugate key
  // from the set has drawbacks (it can be hard to reason about)
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::Conjugate);
  Tensor self_conjugated = self.conj();
  return self_conjugated;
}

void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // Unwrap all arguments
  const auto& arguments = op.schema().arguments();
  const auto num_arguments = arguments.size();
  const auto stack_start = stack->size() - num_arguments;

  // conjugate each tensor argument on the stack with it's conjugate DispatchKey set
  // leave all other arguments unchanged
  for (int64_t i = 0; i < num_arguments; ++i) {
    const auto& ivalue = (*stack)[stack_start + i];
    if (!ivalue.isTensor()) {
      continue;
    }
    if (arguments[i].alias_info().has_value()) {
      TORCH_CHECK(0, "Conjugate fallback doesn't work for input arguments that "
        "alias with outputs; if this is a core op, please report a bug to PyTorch");
    }
    auto* impl = ivalue.unsafeToTensorImpl();
    if (!impl->is_conjugate()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    auto conjugated_tensor = conj_materialize(tensor);
    (*stack)[stack_start + i] = conjugated_tensor;
  }

  op.callBoxed(stack);
}

} // namespace at
