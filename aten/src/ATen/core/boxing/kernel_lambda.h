#pragma once

#include <ATen/core/boxing/kernel_functor.h>
#include <c10/util/TypeTraits.h>

namespace c10 {

namespace detail {
  // WrapRuntimeKernelFunctor: Wraps any runtime functor into a functor that
  // inherits from c10::OperatorKernel, so it can be used as a c10 kernel.
  // This can, for example, be used for lambdas, functors or even function pointers.
  // In the case of function pointers, since it is a runtime function pointer,
  // there is an overhead for calling it whenever the kernel is invoked.
  template<class FuncType, class ReturnType, class ParameterList> class WrapRuntimeKernelFunctor_ {};
  template<class FuncType, class ReturnType, class... Parameters>
  class WrapRuntimeKernelFunctor_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    template<class FuncType_>
    explicit WrapRuntimeKernelFunctor_(FuncType_&& kernel_func)
    : kernel_func_(std::forward<FuncType_>(kernel_func)) {}

    auto operator()(Parameters... args) -> decltype(std::declval<FuncType>()(std::forward<Parameters>(args)...)) {
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

}
