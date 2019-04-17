#pragma once

#include <ATen/core/op_registration/kernel_functor.h>

namespace c10 {
namespace detail {
  // WrapKernelFunction: Wraps a compile time function pointer into a kernel functor.
  // Since it is a compile time function pointer, many compilers can inline it
  // into the wrapper and you don't get any performance overhead for wrapping.
  template<class FuncType, FuncType* kernel_func, class ReturnType, class ParameterList> class WrapKernelFunction_ {};
  template<class FuncType, FuncType* kernel_func, class ReturnType, class... Parameters>
  class WrapKernelFunction_<FuncType, kernel_func, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    auto operator()(Parameters&&... args) -> decltype((*kernel_func)(std::forward<Parameters>(args)...)) {
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

/**
 * Use this to register an operator whose kernel is implemented by a function:
 *
 * Example:
 *
 * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()));
 */
template<class FuncType, FuncType* kernel_func>
inline constexpr auto kernel() ->
// enable_if: only enable it if FuncType is actually a function
guts::enable_if_t<guts::is_function_type<FuncType>::value,
decltype(kernel<typename detail::WrapKernelFunction<FuncType, kernel_func>::type>())> {
  static_assert(!std::is_same<FuncType, KernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");

  return kernel<typename detail::WrapKernelFunction<FuncType, kernel_func>::type>();
}

}
