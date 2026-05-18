#pragma once

#include <c10/core/CompileTimeFunctionPointer.h>

namespace c10::impl {
namespace detail {
template <class FuncPtr, class ReturnType, class ParameterList>
class WrapFunctionIntoFunctor_ {};
template <class FuncPtr, class ReturnType, class... Parameters>
class WrapFunctionIntoFunctor_<
    FuncPtr,
    ReturnType,
    guts::typelist::typelist<Parameters...>>
    final : public c10::OperatorKernel {
 public:
  C10_ALWAYS_INLINE decltype(auto) operator()(Parameters... args) {
    return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
  }
};
} // namespace detail

// WrapFunctionIntoFunctor: Wraps a compile time function pointer into a kernel
// functor. Since it is a compile time function pointer, many compilers can
// inline it into the wrapper and you don't get any performance overhead for
// wrapping.
template <class FuncPtr>
struct WrapFunctionIntoFunctor final {
  static_assert(
      c10::is_compile_time_function_pointer<FuncPtr>::value,
      "WrapFunctionIntoFunctor can only wrap functions created with TORCH_FN.");
  using type = detail::WrapFunctionIntoFunctor_<
      FuncPtr,
      typename guts::function_traits<typename FuncPtr::FuncType>::return_type,
      typename guts::function_traits<
          typename FuncPtr::FuncType>::parameter_types>;
};

} // namespace c10::impl
