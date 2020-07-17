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
using ok_to_box = guts::disjunction<
    std::is_constructible<IValue, T>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, T>,
    // void returns are ok
    std::is_same<void, T>>;

// TODO boxing should be ok for all kernels. Then remove ok_to_box and supports_boxing.

template <typename Result>
using supports_boxing_result =
  guts::negation<guts::disjunction<
    std::is_lvalue_reference<Result>,
    guts::negation<ok_to_box<Result>>,
    std::is_same<IntArrayRef, Result>
  >>;

template <class Result, class... Args>
using supports_boxing =
  guts::conjunction<
    supports_boxing_result<Result>,
    ok_to_box<std::decay_t<Args>>...
  >;

template <typename T, std::enable_if_t<!c10::impl::ok_to_box<T>::value>* = nullptr>
inline bool pushIValueOrCannotBox(std::vector<c10::IValue>& stack, const T& v) {
  torch::jit::push(stack, "cannot box");
  return false;
}
template <typename T, std::enable_if_t<c10::impl::ok_to_box<T>::value>* = nullptr>
inline bool pushIValueOrCannotBox(std::vector<c10::IValue>& stack, const T& v) {
  torch::jit::push(stack, v);
  return true;
}

// boxArgumentsOrCannotBoxIntoStack takes the arguments and pushes them as IValues onto the stack.
// In case the argument cannot be converted to IValue, the function pushes "cannot box"
// IValue string. Return value - whether all of the arguments could be converted to IValues
inline bool boxArgumentsOrCannotBoxIntoStack(std::vector<c10::IValue>& stack) {
  return true;
}
template<typename Item>
inline bool boxArgumentsOrCannotBoxIntoStack(std::vector<c10::IValue>& stack, const Item& item) {
  return pushIValueOrCannotBox(stack, item);
}
template<typename Item, typename... Rest>
inline bool boxArgumentsOrCannotBoxIntoStack(std::vector<c10::IValue>& stack, const Item& item, Rest... other_items) {
  auto res = pushIValueOrCannotBox(stack, item);
  return boxArgumentsOrCannotBoxIntoStack(stack, other_items...) && res;
}

template<class Result>
std::enable_if_t<!supports_boxing_result<Result>::value, Result>
callBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, torch::jit::Stack& stack) {
  TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxedFunc() but return result cannot be boxed");
}

template<class Result>
std::enable_if_t<supports_boxing_result<Result>::value && !std::is_same<void, Result>::value, Result>
callBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, torch::jit::Stack& stack) {
  (*boxed_kernel_func)(functor, opHandle, &stack);
  TORCH_INTERNAL_ASSERT(stack.size() == 1, "A boxed kernel should only push one return to the stack");
  return std::move(stack[0]).to<Result>();
}

template<class Result>
std::enable_if_t<supports_boxing_result<Result>::value && std::is_same<void, Result>::value, Result>
callBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, torch::jit::Stack& stack) {
  (*boxed_kernel_func)(functor, opHandle, &stack);
  TORCH_INTERNAL_ASSERT(stack.size() == 0, "A boxed kernel returned a value but when we called it with KernelFunction::callBoxedFunc(), we expected it to return void.");
}

template<class Result, class... Args>
Result boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args, std::enable_if_t<!supports_boxing<Result, Args...>::value, int> = 0) {
  // Some kernels don't need to actually box, and don't return.  If that's the
  // case, just call them anyway without a stack.  These special cases can be
  // removed once we support boxing everything.
  // See Note [named_not_supported_kernel]
  if (boxed_kernel_func == &named_not_supported_kernel) {
    named_not_supported_kernel(functor, opHandle, nullptr);  // does not return
  }

  TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::call() for a kernel that only has a boxed kernel and doesn't support calling from an unboxed API yet.");
}

// SFINAE version for ops with returns
template<class Result, class... Args>
std::enable_if_t<supports_boxing<Result, Args...>::value && !std::is_same<void, Result>::value, Result>
boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  return callBoxedFunc<Result>(boxed_kernel_func, functor, opHandle, stack);
}

// SFINAE version for ops without returns
template<class Result, class... Args>
std::enable_if_t<supports_boxing<Result, Args...>::value && std::is_same<void, Result>::value, Result>
boxAndCallBoxedFunc(KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, const OperatorHandle& opHandle, Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);

  callBoxedFunc<Result>(boxed_kernel_func, functor, opHandle, stack);
}

}
}
