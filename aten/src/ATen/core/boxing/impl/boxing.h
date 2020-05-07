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
using not_ok_to_box = guts::negation<guts::disjunction<
    std::is_constructible<IValue, T>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, T>,
    // void returns are ok
    std::is_same<void, T>>>;

// TODO boxing should be ok for all kernels. Then remove not_ok_to_box and supports_boxing.

template <class Result, class... Args>
using supports_boxing =
  guts::negation<guts::disjunction<
    std::is_lvalue_reference<Result>,
    not_ok_to_box<Result>,
    std::is_same<IntArrayRef, Result>,
    not_ok_to_box<std::decay_t<Args>>...
  >>;

template<class Result, class... Args>
Result boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args, std::enable_if_t<!supports_boxing<Result, Args...>::value, int> = 0) {
  TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callUnboxed() for a kernel that only has a boxed kernel and doesn't support calling from an unboxed API yet.");
}

// SFINAE version for ops with returns
template<class Result, class... Args>
std::enable_if_t<supports_boxing<Result, Args...>::value && !std::is_same<void, Result>::value, Result>
boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  (*boxed_kernel_func)(functor, opHandle, &stack);

  TORCH_INTERNAL_ASSERT(stack.size() == 1, "A boxed kernel should only push one return to the stack");
  return std::move(stack[0]).to<Result>();
}

// SFINAE version for ops without returns
template<class Result, class... Args>
std::enable_if_t<supports_boxing<Result, Args...>::value && std::is_same<void, Result>::value, Result>
boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  (*boxed_kernel_func)(functor, opHandle, &stack);

  TORCH_INTERNAL_ASSERT(stack.size() == 0, "A boxed kernel returned a value but when we called it with KernelFunction::callUnboxed, we expected it to return void.");
}

}
}
