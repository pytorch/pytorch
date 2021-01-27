#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoFunctor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h>

namespace c10 {

inline KernelFunction::KernelFunction()
: functor_(nullptr)
, boxed_kernel_func_(nullptr)
, unboxed_kernel_func_(nullptr)
{}

inline KernelFunction::KernelFunction(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func)
: functor_(std::move(functor))
, boxed_kernel_func_(boxed_kernel_func)
, unboxed_kernel_func_(unboxed_kernel_func)
{}

template<KernelFunction::BoxedKernelFunction* func>
inline void KernelFunction::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, Stack* stack) {
    func(opHandle, stack);
}

inline bool KernelFunction::isValid() const {
    return boxed_kernel_func_ != nullptr;
}

inline bool KernelFunction::isFallthrough() const {
    return boxed_kernel_func_ == &fallthrough_kernel;
}

inline void KernelFunction::callBoxed(const OperatorHandle& opHandle, Stack* stack) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        boxed_kernel_func_ != nullptr,
        "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction."
    );
    (*boxed_kernel_func_)(functor_.get(), opHandle, stack);
}

template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    return (*func)(functor, std::forward<Args>(args)...);
}

template<class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(const OperatorHandle& opHandle, Args... args) const {
    // note: Args above is intentionally not Args&&. We don't want perfect
    // forwarding, which would require Args to be deduced, but instead we
    // want callers to explicitly specify the Args.

    if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
        return callUnboxedKernelFunction<Return, Args...>(unboxed_kernel_func_, functor_.get(), std::forward<Args>(args)...);
    }

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        boxed_kernel_func_ != nullptr,
        "Tried to call KernelFunction::call() on an uninitialized KernelFunction."
    );

    return impl::BoxedKernelWrapper<Return(Args...)>::call(
        boxed_kernel_func_,
        functor_.get(),
        opHandle,
        std::forward<Args>(args)...
    );
}

template<KernelFunction::BoxedKernelFunction* func>
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

template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
        std::move(kernelFunctor),
        &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        reinterpret_cast<void*>(&impl::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
}

template<class FuncPtr, bool AllowLegacyTypes>
inline KernelFunction KernelFunction::makeFromUnboxedFunction(FuncPtr func_ptr) {
    static_assert(is_compile_time_function_pointer<FuncPtr>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with an invalid parameter. It must be a function pointer created with TORCH_FN.");
    static_assert(!std::is_same<typename FuncPtr::FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    static_assert(FuncPtr::func_ptr() != nullptr, "Kernel function cannot be nullptr");

#if !defined(C10_MOBILE)
    return makeFromUnboxedFunctor<AllowLegacyTypes, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>(
        guts::make_unique_base<OperatorKernel, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>()
    );
#else
    // On mobile, we rather want to optimize for binary size than for performance,
    // so let's not inline the kernel into the wrapper but use makeFromUnboxedRuntimeFunction
    // instead.
    return makeFromUnboxedRuntimeFunction(func_ptr.func_ptr());
#endif
}

template<bool AllowLegacyTypes, class FuncType>
inline KernelFunction KernelFunction::makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(func)
    );
}

template<bool AllowLegacyTypes, class Lambda>
inline std::enable_if_t<guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> KernelFunction::makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

#if !defined(C10_MOBILE)
    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
#else
    // On mobile, we rather want to optimize for binary size than for performance,
    // so let's not inline the kernel into the wrapper but use makeFromUnboxedRuntimeFunction
    // instead.
    using FuncType = typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type;
    return makeFromUnboxedRuntimeFunction<AllowLegacyTypes, FuncType>(lambda);
#endif
}

template<bool AllowLegacyTypes, class Lambda>
inline std::enable_if_t<!guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> KernelFunction::makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
}

}
