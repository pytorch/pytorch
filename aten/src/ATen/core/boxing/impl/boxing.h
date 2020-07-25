#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Dimname.h>

#include <ATen/core/boxing/KernelFunction.h>

#include <c10/util/Metaprogramming.h>

namespace c10 {
namespace impl {

//
// utils
//

// is_mutable_tensor_ref
template<class T> struct is_mutable_tensor_ref : std::false_type {};
template <> struct is_mutable_tensor_ref<at::Tensor&> : std::true_type {};

// is_tuple_of_mutable_tensor_refs
//
template<class T, class Enable = void>
struct is_tuple_of_mutable_tensor_refs : std::false_type {};

template<class T>
struct is_tuple_of_mutable_tensor_refs<T, std::enable_if_t<guts::is_instantiation_of<std::tuple, T>::value, void>>
: guts::typelist::all<is_mutable_tensor_ref, guts::typelist::from_tuple_t<T>>
{};

// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()
//
template<class T, class Enable = void>
struct has_ivalue_to : std::false_type {};

template<class T>
struct has_ivalue_to<T, guts::void_t<decltype(std::declval<IValue>().to<T>())>>
: std::true_type
{};

//
// boxing predicates
//

// A boxable arg type is one that IValue has a constructor for.
template <typename T>
using can_box =
  guts::disjunction<
    std::is_constructible<IValue, std::decay_t<T>>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, std::decay_t<T>>
  >;

template <typename... Ts>
using can_box_all = guts::conjunction<can_box<Ts>...>;

// an unboxable result is one that can be extracted from an IValue
template <typename T>
using can_unbox =
  guts::conjunction<
    guts::disjunction<
      has_ivalue_to<T>,
      // void returns are ok
      std::is_same<void, T>
    >,
    guts::negation<std::is_lvalue_reference<T>>
  >;

//
// profiling support: until boxing support is complete, we push placeholder
// "cannot box" values for unboxable args
//

template <typename T, std::enable_if_t<!c10::impl::can_box<T>::value>* = nullptr>
inline bool pushIValueOrCannotBox(std::vector<c10::IValue>& stack, const T& v) {
  torch::jit::push(stack, "cannot box");
  return false;
}
template <typename T, std::enable_if_t<c10::impl::can_box<T>::value>* = nullptr>
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

//
// BoxedKernelWrapper
//
// For a given function type FT, BoxedKernelWrapper<FT> implements
// a `call` method that
// - takes a boxed kernel and unboxed arguments as specified by FT
// - boxes the arguments
// - calls the boxed kernel
// - unboxes and returns the result
//
// The partial specializations below handle various cases: in
// particular, not all types appearing in op signatures are supported,
// and ops returning references have nonstandard wrapper implementations.
//

// base definition should never be instantiated.
// A "no call method defined on BoxedKernelWrapper" compile error means that
// an op signature has failed to trigger any of the partial specializations
// that follow.
//
template<class FuncType, class Enable = void>
struct BoxedKernelWrapper {
  static_assert(sizeof(FuncType) == -1,
    "Function signature contains one or more unsupported parameter and/or return types. "
    "Look for a nearby error like "
    "\"‘call’ is not a member of ‘c10::impl::BoxedKernelWrapper<(your function type), void>’\" "
    "- (your function type) is the unsupported signature.");
};

// 1. Unsupported type traps.
//
// These specializations capture the remaining gaps in boxing support.
// Rather than triggering compile errors, we generate boxed kernels that
// raise runtime errors. As support for these types is added, the
// specializations can be removed.
//

// at::Quantizer
template <class... Args>
using has_quantizer_arg =
  guts::disjunction<
    std::is_same<at::Quantizer, std::decay_t<Args>>...,
    std::is_same<c10::intrusive_ptr<at::Quantizer>, std::decay_t<Args>>...
  >;

template<class Result, class... Args>
struct BoxedKernelWrapper<Result(Args...), std::enable_if_t<has_quantizer_arg<Args...>::value, void>> {
  static Result call(KernelFunction::InternalBoxedKernelFunction*, OperatorKernel*, const OperatorHandle&, Args... args) {
    TORCH_INTERNAL_ASSERT(false, "Unboxed call to a boxed kernel with unboxable parameter type at::Quantizer.");
  }
};

// 2. Supported signatures, other than ref-passing.
//

template<class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    can_box_all<Args...>::value && can_unbox<Result>::value && !is_tuple_of_mutable_tensor_refs<Result>::value,
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
    stack.reserve(sizeof...(Args));
    torch::jit::push(stack, std::forward<Args>(args)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);

    return guts::if_constexpr<!std::is_same<void, Result>::value>(
      [&] (auto delay_check) {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          stack.size() == 1,
          "Boxed kernel was expected to push exactly one return value to the stack."
        );
        return delay_check(std::move(stack[0]).to<Result>());
      },
      [&] {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          stack.size() == 0,
          "Boxed kernel for op with void return type pushed one or more return values to the stack."
        );
      }
    );
  }
};

// 3. signatures taking a single Tensor reference as their first argument,
// and also returning one.
//
// Note that the passed kernels are assumed to be for inplace/outplace ops,
// and the generated BoxedKernelWrapper specializations will simply return
// the initial argument.
//
template<class... OtherArgs>
struct BoxedKernelWrapper<
  at::Tensor&(at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  static at::Tensor& call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    at::Tensor& outArg,
    OtherArgs... otherArgs
  ) {
    // TODO Reuse stack vector instead of allocating?
    torch::jit::Stack stack;
    stack.reserve(1 + sizeof...(OtherArgs));
    torch::jit::push_one(stack, outArg);
    torch::jit::push(stack, std::forward<OtherArgs>(otherArgs)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, but instead returned ", stack.size(), " values."
    );

    return outArg;
  }
};

// 4. signatures returning a tuple of Tensor references, and taking the same
// number of Tensor refs as their initial arguments.
//
// Note that the passed kernels are assumed to be for inplace/outplace ops,
// and the generated BoxedKernelWrapper specializations will return a tuple
// of those initial arguments.
//
template<class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    can_box_all<Args...>::value && is_tuple_of_mutable_tensor_refs<Result>::value,
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {
    using ArgTuple = std::tuple<Args...>;
    constexpr int RetCount = std::tuple_size<Result>();

    // TODO Reuse stack vector instead of allocating?
    torch::jit::Stack stack;
    stack.reserve(sizeof...(Args));
    torch::jit::push(stack, std::forward<Args>(args)...);

    (*boxed_kernel_func)(functor, opHandle, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, but instead returned ", stack.size(), " values."
    );

    auto result = guts::tuple_take<ArgTuple, RetCount>(ArgTuple{args...});
    static_assert(
        std::is_same<Result, decltype(result)>::value,
        "The parameter list of an op returning a tuple of Tensor references "
            "must begin with an equal number of Tensor reference parameters."
    );
    return result;
  }
};

} // impl
} // c10
