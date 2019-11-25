#include <ATen/core/boxing/boxing.h>

namespace c10 {

inline KernelFunction::KernelFunction()
: functorFactory_()
, functor_(nullptr)
, boxed_kernel_func_(nullptr)
, unboxed_kernel_func_(nullptr)
{}

inline KernelFunction::KernelFunction(std::function<std::unique_ptr<OperatorKernel>()> functorFactory, std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func)
: functorFactory_(std::move(functorFactory))
, functor_(std::move(functor))
, boxed_kernel_func_(boxed_kernel_func)
, unboxed_kernel_func_(unboxed_kernel_func)
{}

template<KernelFunction::BoxedKernelFunction* func>
inline void KernelFunction::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, Stack* stack) {
    func(opHandle, stack);
}

inline OperatorKernel* KernelFunction::getFunctor_() const {
    if (functor_.get() == nullptr) {
        if (!functorFactory_) {
        return nullptr;
        }
        functor_ = functorFactory_();
    }
    return functor_.get();
}


inline bool KernelFunction::isValid() const {
    // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this should only check boxed_kernel_func_.
    return boxed_kernel_func_ != nullptr || unboxed_kernel_func_ != nullptr;
}

inline void KernelFunction::callBoxed(const OperatorHandle& opHandle, Stack* stack) const {
    if (C10_UNLIKELY(boxed_kernel_func_ == nullptr)) {
        if (unboxed_kernel_func_ == nullptr) {
            TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction.");
        } else {
            // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this case should be impossible.
            TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed().");
        }
    }

    (*boxed_kernel_func_)(getFunctor_(), opHandle, stack);
}

template<class Return, class... Args>
inline Return KernelFunction::callUnboxed(const OperatorHandle& opHandle, Args... args) const {
    // note: Args above is intentionally not Args&&. We don't want perfect
    // forwarding, which would require Args to be deduced, but instead we
    // want callers to explicitly specify the Args.

    if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
        using ActualSignature = Return (OperatorKernel*, Args...);
        ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func_);
        return (*func)(getFunctor_(), std::forward<Args>(args)...);
    }

    TORCH_INTERNAL_ASSERT(boxed_kernel_func_ != nullptr, "Tried to call KernelFunction::callUnboxed() on an uninitialized KernelFunction.");
    return impl::boxAndCallBoxedFunc<Return, Args...>(boxed_kernel_func_, getFunctor_(), opHandle, std::forward<Args>(args)...);
}

template<KernelFunction::BoxedKernelFunction* func>
inline KernelFunction KernelFunction::makeFromBoxedFunction() {
    return KernelFunction(
        nullptr,  // no functorFactory_, this can only be called in a boxed way.
        nullptr,  // no functor_ object either
        &make_boxed_function<func>,
        nullptr  // no unboxed function pointer
    );
}

template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
        nullptr, // no functorFactory_ because we already have the functor_
        std::move(kernelFunctor),
        &detail::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
}

template<class KernelFunctor, bool AllowLegacyTypes>
inline KernelFunction KernelFunction::makeFromUnboxedFunctorFactory(std::function<std::unique_ptr<OperatorKernel>()> kernelFunctorFactory) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
        std::move(kernelFunctorFactory),
        nullptr, // delay creation of functor_ (it will be created by calling functorFactory_ later)
        &detail::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
}

template<class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedOnlyFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
        nullptr, // no functorFactory_ because we already have the functor_
        std::move(kernelFunctor),
        nullptr, // Don't create a boxed kernel for this
        reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
}

template<class FuncType, FuncType* func, bool AllowLegacyTypes>
inline KernelFunction KernelFunction::makeFromUnboxedFunction() {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with invalid template parameters. They must be <FuncType, *func_ptr>.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes, typename detail::WrapKernelFunction<FuncType, func>::type>(
        guts::make_unique_base<OperatorKernel, typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
}

template<class FuncType, FuncType* func>
inline KernelFunction KernelFunction::makeFromUnboxedOnlyFunction() {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedOnlyFunction with invalid template parameters. They must be <FuncType, *func_ptr>.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedOnlyFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedOnlyFunctor<typename detail::WrapKernelFunction<FuncType, func>::type> (
        guts::make_unique_base<OperatorKernel, typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
}

template<bool AllowLegacyTypes, class FuncType>
inline KernelFunction KernelFunction::makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes, detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(
        guts::make_unique_base<OperatorKernel, detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(func)
    );
}

template<class FuncType>
inline KernelFunction KernelFunction::makeFromUnboxedOnlyRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedOnlyFunctor<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(
        guts::make_unique_base<OperatorKernel, detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(func)
    );
}

template<bool AllowLegacyTypes, class Lambda>
inline KernelFunction KernelFunction::makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<guts::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    return makeFromUnboxedFunctor<AllowLegacyTypes, detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
}

}
