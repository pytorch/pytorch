#pragma once

#include <ATen/core/boxing/kernel_functor.h>

namespace c10 {
namespace detail {
  // WrapKernelFunction: Wraps a compile time function pointer into a kernel functor.
  // Since it is a compile time function pointer, many compilers can inline it
  // into the wrapper and you don't get any performance overhead for wrapping.
  template<class FuncType, FuncType* kernel_func, class ReturnType, class ParameterList> class WrapKernelFunction_ {};
  template<class FuncType, FuncType* kernel_func, class ReturnType, class... Parameters>
  class WrapKernelFunction_<FuncType, kernel_func, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    auto operator()(Parameters... args) -> decltype((*kernel_func)(std::forward<Parameters>(args)...)) {
      return (*kernel_func)(std::forward<Parameters>(args)...);
    }
  };
  template<class FuncType, FuncType* kernel_func, class Enable = guts::enable_if_t<guts::is_function_type<FuncType>::value>>
  struct WrapKernelFunction final {
    using type = WrapKernelFunction_<
        FuncType,
        kernel_func,
        typename guts::function_traits<FuncType>::return_type,
        typename guts::function_traits<FuncType>::parameter_types
    >;
  };
}

}
