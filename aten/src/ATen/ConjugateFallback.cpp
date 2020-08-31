#include <ATen/ConjugateFallback.h>
#include <ATen/native/UnaryOps.h>

namespace at {

    TORCH_LIBRARY_IMPL(_, Conjugate, m) {
        m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
    }

   Tensor conj_materialize(const Tensor& self) {
        // this function assumes that the tensor input has it's conjugate bit set
        self.set_conjugate(false);
        Tensor self_conjugated = self.conj();
        return self_conjugated;
    }

    void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
        // Unwrap all arguments
        const auto num_arguments = op.schema().arguments().size();
        auto arguments = torch::jit::pop(*stack, num_arguments);

        // conjugate each tensor argument on the stack with it's conjugate DispatchKey + bit set
        // leave all other arguments unchanged
        for (int64_t idx = 0; idx < arguments.size(); ++idx) {
            const auto& ivalue = arguments[idx];
            if (!ivalue.isTensor()) {
                torch::jit::push(stack, ivalue);
                continue;
            }
            auto* impl = ivalue.unsafeToTensorImpl();
            if (impl->is_conjugate()) {
                const auto& tensor = ivalue.toTensor();
                auto conjugated_tensor = conj_materialize(tensor);
                torch::jit::push(stack, conjugated_tensor);
            } else {
                torch::jit::push(stack, ivalue);
            }
        }

        op.callBoxed(stack);
    }
}
