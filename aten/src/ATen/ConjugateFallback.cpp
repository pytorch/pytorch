#include <ATen/ConjugateFallback.h>
#include <ATen/native/UnaryOps.h>

namespace at {

    TORCH_LIBRARY_IMPL(_, Conjugate, m) {
        m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
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
        const auto num_arguments = op.schema().arguments().size();

        // conjugate each tensor argument on the stack with it's conjugate DispatchKey set
        // leave all other arguments unchanged
        for (int64_t idx = stack->size() - num_arguments; idx < stack->size(); ++idx) {
            const auto& ivalue = (*stack)[idx];
            if (!ivalue.isTensor()) {
                continue;
            }
            auto* impl = ivalue.unsafeToTensorImpl();
            if (!impl->is_conjugate()) {
                continue;
            }
            const auto& tensor = ivalue.toTensor();
            auto conjugated_tensor = conj_materialize(tensor);
            (*stack)[idx] = conjugated_tensor;
        }

        op.callBoxed(stack);
    }
}
