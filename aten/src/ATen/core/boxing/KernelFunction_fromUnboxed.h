#pragma once
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoFunctor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h>

namespace c10 {
namespace kernel_function {

using BoxedKernelFunction = void(const OperatorHandle&, Stack*);

template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
#ifndef NDEBUG
  // This assertion is costly for build time so it's debug-gated.
  static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call kernel_function::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
#endif
  static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call kernel_function::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

  return KernelFunction(
      std::move(kernelFunctor),
      &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
      reinterpret_cast<void*>(&impl::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
}

template<class FuncPtr, bool AllowLegacyTypes>
inline KernelFunction makeFromUnboxedFunction(FuncPtr func_ptr) {
    static_assert(is_compile_time_function_pointer<FuncPtr>::value, "Tried to call kernel_function::makeFromUnboxedFunction with an invalid parameter. It must be a function pointer created with TORCH_FN.");
    static_assert(!std::is_same<typename FuncPtr::FuncType, BoxedKernelFunction>::value, "Tried to call kernel_function::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
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

/**
 * Create a KernelFunction from an unboxed function.
 * KernelFunction::makeFromUnboxedFunction is usually a better choice than
 * this if you know the function pointer at compile time, see doc comment
 * there for an explanation.
 *
 * Example:
 *
 * > Tensor unboxed_func(Tensor a, Tensor b) {...}
 * > KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&unboxed_func);
 */
template<bool AllowLegacyTypes=false, class FuncType>
inline KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call kernel_function::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call kernel_function::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(func)
    );
}


/**
 * Create a KernelFunction from an unboxed lambda.
 *
 * Example:
 *
 * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
 * >      [] (Tensor a, bool b) -> Tensor {...});
 */
template<bool AllowLegacyTypes=false, class Lambda>
inline std::enable_if_t<guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call kernel_function::makeFromUnboxedLambda with a non-lambda type.");

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

template<bool AllowLegacyTypes=false, class Lambda>
inline std::enable_if_t<!guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call kernel_function::makeFromUnboxedLambda with a non-lambda type.");

    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
}

}}  // namespace c10::kernel_function
