#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <c10/core/TensorOptions.h>

#include <ATen/core/boxing/KernelFunction.h>

#include <c10/util/Metaprogramming.h>

namespace c10 {
namespace impl {

//
// utils
//

// is_mutable_tensor_ref
template <class T> struct is_mutable_tensor_ref : std::false_type {};
template <> struct is_mutable_tensor_ref<at::Tensor&> : std::true_type {};

// is_tuple_of_mutable_tensor_refs
//
template <class T, class Enable = void>
struct is_tuple_of_mutable_tensor_refs : std::false_type {};

template <class T>
struct is_tuple_of_mutable_tensor_refs<T, std::enable_if_t<guts::is_instantiation_of<std::tuple, T>::value, void>>
: guts::typelist::all<is_mutable_tensor_ref, guts::typelist::from_tuple_t<T>>
{};

// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()
//
template <class T, class Enable = void>
struct has_ivalue_to : std::false_type {};

template <class T>
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
// boxArgs - utility for pushing unboxed args onto IValue stack
//
template <class... Args>
static torch::jit::Stack boxArgs(Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  stack.reserve(sizeof...(Args));
  torch::jit::push(stack, std::forward<Args>(args)...);
  return stack;
}

//
// PopResult is a helper class whose specializations handle popping single and
// multiple return values, respectively.
//
template <class Result>
struct PopResult final {
  static Result call(Stack& stack) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return one value on the stack, ",
      "but instead pushed ", stack.size(), " values."
    );
    return std::move(stack[0]).to<Result>();
  }
};

template <class... Types>
struct PopResult<std::tuple<Types...>> final {
  using Result = std::tuple<Types...>;

  static Result call(Stack& stack) {
    // for tuple return types, boxed kernel has pushed multiple values onto the stack
    constexpr int RetCount = sizeof...(Types);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == RetCount,
      "Boxed kernel was expected to return ", RetCount, " values on the stack, ",
      "but instead pushed ", stack.size(), " values."
    );
    return pop_to_tuple_impl(stack, std::make_index_sequence<RetCount>());
  }
private:
  // note: this has been moved into its own helper only to avoid a parse error on `indices` otherwise.
  // I'm sure there's an incantation that slips it past the parser but eh
  template <size_t... indices>
  static Result pop_to_tuple_impl(Stack& stack, std::index_sequence<indices...>) {
    return std::make_tuple((std::move(stack[indices]).to<Types>())...);
  }
};

//
// BoxedKernelWrapper
//
// For a given function type FT, BoxedKernelWrapper<FT> implements
// a `call` method that
// - takes a boxed kernel and unboxed arguments as specified by FT,
// - calls `boxArgs` to box the arguments
// - calls the boxed kernel
// - unboxes and returns the result
//
// The partial specializations below handle various cases: in
// particular, not all types appearing in op signatures are supported,
// and ops returning references have nonstandard wrapper implementations.
//

// 1. The base specialization of BoxedKernelWrapper should never be instantiated.
// A "no call method defined on BoxedKernelWrapper" compile error means that
// an op signature has failed to trigger any of the partial specializations
// that follow this one.
//
template <class FuncType, class Enable = void>
struct BoxedKernelWrapper {
  // The reason we're not just doing straight up static_assert(false, ...) here:
  // Basically, the way to make sure a static_assert only fires if a template
  // is actually instantiated (rather than every time the file is parsed) is to use
  // template parameters in the expression, e.g. FuncType here. However, since
  // `sizeof(FuncType) != sizeof(FuncType)` is always false, this has the same
  // effect.
  static_assert(sizeof(FuncType) != sizeof(FuncType),
     "Function signature contains one or more unsupported parameter and/or return types. "
     "Look for a nearby error like "
     "\"'call' is not a member of 'c10::impl::BoxedKernelWrapper<(your function type), void>'\" "
     "- (your function type) is the unsupported signature.");
};

//
// 2. Supported signatures, other than those involving non-const Tensor refs -
// i.e., "functional" ops.
//

template <class Result, class... Args>
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
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    torch::jit::Stack stack = boxArgs(args...);
    (*boxed_kernel_func)(functor, opHandle, dispatchKeySet, &stack);

    return guts::if_constexpr<!std::is_same<void, Result>::value>(
      [&] (auto delay_check) {
        // op has pushed one or more values onto the stack.
        return delay_check(PopResult<Result>::call(stack));
      },
      [&] {
        // op returns void, boxed kernel has pushed nothing onto stack.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          stack.size() == 0,
          "Boxed kernel was expected to return no values on the stack, ",
          "but instead returned ", stack.size(), " values."
        );
      }
    );
  }
};

//
// 3. in-place ops take a single non-const Tensor reference
// as their first argument, and return it.
//
// Note: all signatures matching this pattern are are assumed to be for such ops.
// Because of this, the generated BoxedKernelWrapper specializations simply
// return the in-place argument.
//

template <class... OtherArgs>
struct BoxedKernelWrapper<
  at::Tensor&(at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  static at::Tensor& call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    at::Tensor& outArg, OtherArgs... otherArgs
  ) {
    torch::jit::Stack stack = boxArgs(outArg, otherArgs...);
    (*boxed_kernel_func)(functor, opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    return outArg;
  }
};

//
// 4. out of place ops that take a single non-const Tensor reference as their
// final argument, and also return it.
//
// Note: all signatures matching this pattern are are assumed to be for such ops.
// This assumption permits the generated BoxedKernelWrapper specializations to simply
// return out arguments.
//
template <class FirstArg, class... RestArgs>
struct BoxedKernelWrapper<
  at::Tensor&(FirstArg, RestArgs...),
  std::enable_if_t<
    can_box_all<FirstArg, RestArgs...>::value
    // this skips over in-place kernels with a non-const Tensor
    // arg at the front, so those can unambiguously trigger the preceding specialization.
    && !is_mutable_tensor_ref<FirstArg>::value,
    void
  >
> {
  static at::Tensor& call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    FirstArg firstArg, RestArgs... restArgs
  ) {
    torch::jit::Stack stack = boxArgs(firstArg, restArgs...);
    (*boxed_kernel_func)(functor, opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    return std::get<sizeof...(RestArgs) - 1>(std::tuple<RestArgs...>{restArgs...});
  }
};

//
// 5. out of place ops that take multiple non-const Tensor references as their
// final arguments, and return them in a std::tuple.
//
// Note: all signatures matching this pattern are are assumed to be for such ops.
// This assumption permits the generated BoxedKernelWrapper specializations to simply
// return the out arguments.
//
template <class Result, class... Args>
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
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    using ArgTuple = std::tuple<Args...>;
    constexpr int RetCount = std::tuple_size<Result>();

    torch::jit::Stack stack = boxArgs(args...);
    (*boxed_kernel_func)(functor, opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == RetCount,
      "Boxed kernel was expected to return ", RetCount, " values on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    auto result = guts::tuple_take<ArgTuple, -RetCount>(ArgTuple{args...});
    static_assert(
        std::is_same<Result, decltype(result)>::value,
        "The parameter list of an op returning a tuple of Tensor references "
            "must end with an equal number of Tensor reference parameters."
    );
    return result;
  }
};

} // impl
} // c10
