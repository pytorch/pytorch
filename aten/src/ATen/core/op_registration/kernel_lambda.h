#pragma once

#include <ATen/core/op_registration/kernel_functor.h>
#include <c10/util/TypeTraits.h>

namespace c10 {

namespace detail {
  // WrapRuntimeKernelFunctor: Wraps any runtime functor into a functor that
  // inherits from c10::OperatorKernel, so it can be used as a c10 kernel.
  // This can, for example, be used for lamdas, functors or even function pointers.
  // In the case of function pointers, since it is a runtime function pointer,
  // there is an overhead for calling it whenever the kernel is invoked.
  template<class FuncType, class ReturnType, class ParameterList> class WrapRuntimeKernelFunctor_ {};
  template<class FuncType, class ReturnType, class... Parameters>
  class WrapRuntimeKernelFunctor_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    template<class FuncType_>
    explicit WrapRuntimeKernelFunctor_(FuncType_&& kernel_func)
    : kernel_func_(std::forward<FuncType_>(kernel_func)) {}

    auto operator()(Parameters&&... args) -> decltype(std::declval<FuncType>()(std::forward<Parameters>(args)...)) {
      return kernel_func_(std::forward<Parameters>(args)...);
    }

  private:
    FuncType kernel_func_;
  };
  template<class FuncType>
  using WrapRuntimeKernelFunctor = WrapRuntimeKernelFunctor_<
      FuncType,
      typename guts::infer_function_traits_t<FuncType>::return_type,
      typename guts::infer_function_traits_t<FuncType>::parameter_types
  >;
}

/**
 * Use this to register an operator whose kernel is implemented as a stateless lambda.
 *
 * Example:
 *
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel([] (Tensor a) -> Tensor{...}),
 * >         c10::dispatchKey(CPUTensorId()));
 */
template<class Lambda>
inline constexpr auto kernel(Lambda&& functor) ->
// enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
guts::enable_if_t<guts::is_functor<guts::decay_t<Lambda>>::value,
decltype(detail::kernelFunctor<detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(functor)))> {
  static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

  // We don't support stateful lambdas (i.e. lambdas with a capture), because their
  // behavior would be nonobvious. A functor kernel with cache gets a new instance of
  // its cache each time the kernel is looked up from the dispatch table.
  // A lambda with a capture would be global and share its capture between all kernel lookups.
  // So, instead of making users having to think about it (including the thread-safety
  // issues this causes), let's just forbid stateful lambdas alltogether.
  static_assert(guts::is_stateless_lambda<guts::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

  return detail::kernelFunctor<detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(functor));
}

}
