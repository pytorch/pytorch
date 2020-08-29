#include <ATen/ConjugateFallback.h>
#include <ATen/native/UnaryOps.h>

namespace at {

    TORCH_LIBRARY_IMPL(_, Conjugate, m) {
        m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
    }

    void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
        // Unwrap all arguments
        const auto num_arguments = op.schema().arguments().size();
        const auto arguments = torch::jit::last(stack, num_arguments);
        auto args = torch::jit::pop(*stack, num_arguments);
        std::cout << "calling conj fallback" << std::endl;
        // c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::Conjugate);

        // conjugate each tensor argument on the stack
        // with it's conjugate DispatchKey + bit set
        // leave all other arguments unchanged
        for (int64_t idx = 0; idx < arguments.size(); ++idx) {
            const auto& ivalue = arguments[idx];
            if (!ivalue.isTensor()) {
                std::cout << "case 0: not a tensor" << std::endl;
                torch::jit::push(stack, ivalue);
                continue;
            }
            auto* impl = args[idx].unsafeToTensorImpl();
            if (impl->is_conjugate()) {
                const auto& tensor = ivalue.toTensor();
                // TODO (delete) my understanding so far:
                // I can think of conjugation as an intermediate function in the dispatch chain like so:
                // args: [arg1 (not tensor), arg2 (tensor with it's conj bit set)]
                // returns: [arg1 (unchanged), arg2 (newly conjugated tensor)]
                auto conjugated_tensor = tensor.conj_materialize();
                std::cout << "case 1: performed conjugation! is conj bit still set? " << conjugated_tensor.is_conjugate() << std::endl;
                torch::jit::push(stack, conjugated_tensor);
                // IValue ivalue_tensor = IValue(conjugated_tensor);
                // arguments[idx] = ivalue_tensor;
            } else {
                std::cout << "case 2: tensor has no conj bit set" << std::endl;
                torch::jit::push(stack, ivalue);
            }
        }

        // for (const auto& arg : args) {
            // torch::jit::push(stack, arg);
        // }

        op.callBoxed(stack);
    }
}
