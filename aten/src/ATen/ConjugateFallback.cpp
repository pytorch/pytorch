#include <ATen/ConjugateFallback.h>
#include <ATen/native/UnaryOps.h>

namespace at {
    void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
        // Unwrap all arguments
        const auto num_arguments = op.schema().arguments().size();
        const auto arguments = torch::jit::last(stack, num_arguments);
        auto args = torch::jit::pop(*stack, num_arguments);

        for (int64_t idx = 0; idx < arguments.size(); ++idx) {
            const auto& ivalue = arguments[idx];
            if (!ivalue.isTensor()) {
                continue;
            }
            auto* impl = args[idx].unsafeToTensorImpl();
            if (impl->key_set().has(DispatchKey::Conjugate)) {
                const auto& tensor = ivalue.toTensor();
                // TODO: I think I want to take the existing tensor on the stack
                // and run conjugation on it.
                // The existing conj() function creates a new Tensor.
                // Is it reaonsable to set the memory on the stack in place like this?
                // I don't see other examples of fallbacks doing this,
                // so instead I'm pushing the new conjugated vector onto the stack
                auto conjugated_tensor = tensor.conj();
                torch::jit::push(stack, conjugated_tensor);
            }
        }

        op.callBoxed(stack);
    }
}
