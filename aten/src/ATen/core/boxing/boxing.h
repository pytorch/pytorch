#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/boxing/KernelFunction.h>

namespace at {
struct Dimname;
}

namespace c10 {
namespace impl {

// Assume T is decayed
template <typename T>
using not_ok_to_box =
  c10::guts::disjunction<
    c10::guts::negation<
      c10::guts::disjunction<
        std::is_constructible<IValue, T>,
        // TensorOptions are not directly constructible into IValue,
        // but torch::jit::push knows how to handle them
        std::is_same<TensorOptions, T>,
        // void returns are ok
        std::is_same<void, T>
      >>
#ifdef BUILD_NAMEDTENSOR
    ,
    // some constructors are templated (and therefore pass
    // is_constructible), but do not actually work with all
    // template arguments, so we must blacklist them explicitly
    // TODO: The correct fix is to sfinae based on is_constructible of T
    std::is_same<optional<ArrayRef<at::Dimname>>, T>
#endif
  >;

// TODO boxing should be ok for all kernels. Then remove not_ok_to_box and supports_boxing.

template <class Result, class... Args>
using supports_boxing =
  c10::guts::negation<c10::guts::disjunction<
    std::is_lvalue_reference<Result>,
    not_ok_to_box<Result>,
    std::is_same<IntArrayRef, Result>,
    not_ok_to_box<guts::decay_t<Args>>...
  >>;

template<class Result, class... Args>
Result boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args, guts::enable_if_t<!supports_boxing<Result, Args...>::value, int> = 0) {
  TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callUnboxed() for a kernel that only has a boxed kernel and doesn't support calling from an unboxed API yet.");
}

template<class Result, class... Args>
Result boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args, guts::enable_if_t<supports_boxing<Result, Args...>::value && !std::is_same<void, Result>::value, int> = 0) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  (*boxed_kernel_func)(functor, opHandle, &stack);

  TORCH_INTERNAL_ASSERT(stack.size() == 1, "A boxed kernel should only push one return to the stack");
  return std::move(stack[0]).to<Result>();
}

template<class Result, class... Args>
Result boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args, guts::enable_if_t<supports_boxing<Result, Args...>::value && std::is_same<void, Result>::value, int> = 0) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  (*boxed_kernel_func)(functor, opHandle, &stack);

  TORCH_INTERNAL_ASSERT(stack.size() == 0, "A boxed kernel returned a value but when we called it with KernelFunction::callUnboxed, we expected it to return void.");
}

}
}
