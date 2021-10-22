#pragma once
#include <ATen/core/boxing/impl/WrapFunctionIntoFunctor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h>

namespace c10 {

inline KernelFunction::KernelFunction()
    : functor_()
, boxed_kernel_func_(nullptr)
, unboxed_kernel_func_(nullptr)
{}

inline KernelFunction::KernelFunction(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func)
: functor_(std::move(functor))
, boxed_kernel_func_(boxed_kernel_func)
, unboxed_kernel_func_(unboxed_kernel_func)
{}

template<KernelFunction::BoxedKernelFunction* func>
inline void KernelFunction::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet, Stack* stack) {
    // Note that we're dropping the DispatchKeySet argument.
    // See Note [Plumbing Keys Through The Dispatcher 2] for details.
    func(opHandle, stack);
}

inline bool KernelFunction::isValidUnboxed() const {
    return unboxed_kernel_func_ != nullptr;
}

template<KernelFunction::BoxedKernelFunction_withDispatchKeys* func>
inline void KernelFunction::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet ks, Stack* stack) {
    // See Note [Plumbing Keys Through The Dispatcher 2] for details.
    func(opHandle, ks, stack);
}

inline bool KernelFunction::isValid() const {
    return boxed_kernel_func_ != nullptr;
}

inline bool KernelFunction::isFallthrough() const {
    return boxed_kernel_func_ == &fallthrough_kernel;
}

inline void KernelFunction::callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        boxed_kernel_func_ != nullptr,
        "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction."
    );
    (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
}

template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, DispatchKeySet dispatchKeySet, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...);
}

template<KernelFunction::BoxedKernelFunction* func>
inline KernelFunction KernelFunction::makeFromBoxedFunction() {
    return KernelFunction(
        nullptr,  // no functor_ object
        &make_boxed_function<func>,
        nullptr  // no unboxed function pointer
    );
}

template<KernelFunction::BoxedKernelFunction_withDispatchKeys* func>
inline KernelFunction KernelFunction::makeFromBoxedFunction() {
    return KernelFunction(
        nullptr,  // no functor_ object
        &make_boxed_function<func>,
        nullptr  // no unboxed function pointer
    );
}

inline KernelFunction KernelFunction::makeFallthrough() {
    return KernelFunction(
        nullptr,  // no functor_ object
        &fallthrough_kernel,
        nullptr  // no unboxed function pointer
    );
}

inline KernelFunction KernelFunction::makeAmbiguousAutogradOther() {
    return KernelFunction(
        nullptr,  // no functor_ object
        &ambiguous_autogradother_kernel,
        nullptr  // no unboxed function pointer
    );
}

inline KernelFunction KernelFunction::makeNamedNotSupported() {
    return KernelFunction(
        nullptr,  // no functor_ object
        &named_not_supported_kernel,
        nullptr  // no unboxed function pointer
    );
}

template<class KernelFunctor>
inline KernelFunction KernelFunction::makeFromBoxedFunctor(std::unique_ptr<KernelFunctor> kernelFunctor) {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromBoxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
    return KernelFunction(
        std::move(kernelFunctor),
        [](OperatorKernel* kernel, const OperatorHandle& op, DispatchKeySet ks, Stack* stack) {
          (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
        },
        nullptr // no unboxed function pointer
    );
}

}
