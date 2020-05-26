#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/boxing/KernelFunction.h>

#include <ATen/core/Dimname.h>


namespace c10 {
namespace impl {

// Assume T is decayed
template <typename T>
using boxable_arg =
  guts::disjunction<
    std::is_constructible<IValue, T>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, T>
  >;

template <typename... Args>
using boxable_args =
  guts::conjunction<
    boxable_arg<std::decay_t<Args>>...
  >;

template <typename T>
using boxable_result =
  guts::conjunction<
    guts::disjunction<
      // using IValue(T) as a proxy for IValue.to<T>() here
      std::is_constructible<IValue, T>,
      // void returns are ok
      std::is_same<void, T>
    >,
    guts::negation<std::is_same<IntArrayRef, T>>
  >;

template <class Result, class... Args>
using supports_boxing =
  guts::conjunction<
    boxable_args<Args...>,
    boxable_result<Result>
  >;

// ---

template <class... Args>
using uses_dimname =
  guts::disjunction<
    std::is_same<at::Dimname, std::decay_t<Args>>...,
    std::is_same<c10::ArrayRef<at::Dimname>, std::decay_t<Args>>...,
    std::is_same<c10::optional<c10::ArrayRef<at::Dimname>>, std::decay_t<Args>>...
  >;

template <class... Args>
using uses_quantizer =
  guts::disjunction<
    std::is_same<at::Quantizer, std::decay_t<Args>>...,
    std::is_same<c10::intrusive_ptr<at::Quantizer>, std::decay_t<Args>>...
  >;

// ---

// base definition establishes template arity.
// should never be instantiated - the partial specializations that
// follow should cover all FuncType instances, both supported and
// unsupported. "no call method defined on BoxAndCallBoxedFunc"
// errors are a sign that this coverage is incomplete.
//
template<class FuncType, class Enable = void>
struct BoxAndCallBoxedFunc {};

template<class Result, class... Args>
struct BoxAndCallBoxedFunc<
  Result(Args...),
  std::enable_if_t<
    !supports_boxing<Result, Args...>::value && uses_dimname<Args...>::value,
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {
    TORCH_INTERNAL_ASSERT(
      false,
      "Tried to call KernelFunction::call() for a kernel that uses at::Dimname, "
      "which is not supported when calling from an unboxed API."
    );
  }
};

template<class Result, class... Args>
struct BoxAndCallBoxedFunc<
  Result(Args...),
  std::enable_if_t<
    !supports_boxing<Result, Args...>::value && uses_quantizer<Args...>::value,
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {
    TORCH_INTERNAL_ASSERT(
      false,
      "Tried to call KernelFunction::call() for a kernel that uses at::Quantizer, "
      "which is not yet supported when calling from an unboxed API."
    );
  }
};

// supported signatures generate instances of this definition, with
// the exception of ops that return (one or more) references. These
// must be handled differently due to ownership issues - see partial
// specializations that follow.
//
template<class Result, class... Args>
struct BoxAndCallBoxedFunc<
  Result(Args...),
  std::enable_if_t<
    supports_boxing<Result, Args...>::value &&
      !std::is_lvalue_reference<Result>::value,
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {
    // TODO Reuse stack vector instead of allocating?
    torch::jit::Stack stack;
    torch::jit::push(stack, std::forward<Args>(args)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);

    return guts::if_constexpr<!std::is_same<void, Result>::value>(
      [&] (auto delay_check) {
        TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Boxed kernel was expected to push exactly one return to the stack."
        );
        return delay_check(std::move(stack[0]).to<Result>());
      },
      [&] {
        TORCH_INTERNAL_ASSERT(
          stack.size() == 0,
          "Boxed kernel for op with void return type pushed one or more "
          "return values to the stack."
        );
      }
    );
  }
};

// a signature returning a reference of the same type as its initial
// argument is assumed to return a reference to that argument (e.g.
// a method that returns a self-reference, or a function that takes
// and returns an out argument)
//
template<class Result, class... OtherArgs>
struct BoxAndCallBoxedFunc<
  Result&(Result&, OtherArgs...),
  std::enable_if_t<boxable_args<OtherArgs...>::value, void>
> {
  static Result& call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Result& outArg,
    OtherArgs... otherArgs
  ) {
    // TODO Reuse stack vector instead of allocating?
    torch::jit::Stack stack;
    torch::jit::push_one(stack, outArg);
    torch::jit::push(stack, std::forward<OtherArgs>(otherArgs)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);

    TORCH_INTERNAL_ASSERT(
      stack.size() == 1,
      "Boxed kernel was expected to push exactly one return value to the stack."
    );

    return outArg;
  }
};

// a signature returning a tuple of references, of the same type as
// the equivalent number of initial arguments, is assumed to return
// references to those arguments (e.g., a function that takes and
// returns a tuple of out arguments).
//
template<class... Results, class... Args>
struct BoxAndCallBoxedFunc<
  std::tuple<Results...>(Args...),
  std::enable_if_t<
    boxable_args<Args...>::value &&
      guts::conjunction<std::is_lvalue_reference<Results>...>::value,
    void
  >
> {
  using ResultsTuple = std::tuple<Results...>;

  static ResultsTuple call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {

    TORCH_INTERNAL_ASSERT(
      false,
      "Tried to call KernelFunction::call() for a kernel that returns multiple "
      "out args, which is not supported when calling from an unboxed API."
    );

#if 0
    // TODO Reuse stack vector instead of allocating?
    torch::jit::Stack stack;
    torch::jit::push(stack, std::forward<Args>(args)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);

    constexpr size_t num_results = std::tuple_size<ResultsTuple>::value;
    TORCH_INTERNAL_ASSERT(
      stack.size() == num_results,
      "Boxed kernel pushed incorrect number of return values."
    );

    return ResultsTuple(args...);
#endif
  }
};

} // impl
} // c10
