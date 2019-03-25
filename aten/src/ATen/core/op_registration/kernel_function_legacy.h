#pragma once

#include <ATen/core/op_registration/kernel_functor.h>

namespace c10 {
namespace detail {
  // WrapKernelFunctionRuntime: Wraps a runtime function pointer into a kernel functor.
  // Since it is a runtime function pointer, there is an overhead for calling
  // the function pointer whenever the kernel is invoked.
  template<class FuncType, class ReturnType, class ParameterList> class WrapKernelFunctionRuntime_ {};
  template<class FuncType, class ReturnType, class... Parameters>
  class WrapKernelFunctionRuntime_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    explicit WrapKernelFunctionRuntime_(FuncType* kernel_func)
    : kernel_func_(kernel_func) {}

    auto operator()(Parameters&&... args) -> decltype(std::declval<FuncType>()(std::forward<Parameters>(args)...)) {
      return (*kernel_func_)(std::forward<Parameters>(args)...);
    }

  private:
    FuncType* kernel_func_;
 } ;
  template<class FuncType>
  using WrapKernelFunctionRuntime = WrapKernelFunctionRuntime_<
      FuncType,
      typename guts::function_traits<FuncType>::return_type,
      typename guts::function_traits<FuncType>::parameter_types
  >;
}

}
