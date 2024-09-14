#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/core/TensorOptions.h>

#include <ATen/core/boxing/BoxedKernel.h>

#include <c10/util/Metaprogramming.h>
#include <type_traits>

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
struct ivalue_to_helper
{
    using type = decltype(std::declval<IValue>().template to<T>());
};
template <class T>
using ivalue_to_helper_t = typename ivalue_to_helper<T>::type;

template <class T>
struct has_ivalue_to<T, std::void_t<ivalue_to_helper_t<T>>>
: std::true_type
{};

//
// boxing predicates
//

// A boxable arg type is one that IValue has a constructor for.
template <typename T>
using can_box =
  std::disjunction<
    std::is_constructible<IValue, std::decay_t<T>>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, std::decay_t<T>>
  >;

template <typename... Ts>
using can_box_all = std::conjunction<can_box<Ts>...>;

// an unboxable result is one that can be extracted from an IValue
template <typename T>
using can_unbox =
   std::conjunction<
    std::disjunction<
      has_ivalue_to<T>,
      // void returns are ok
      std::is_same<void, T>
    >,
    std::negation<std::is_lvalue_reference<T>>
  >;

//
// boxArgs - utility for pushing unboxed args onto IValue stack
//
template <class... Args>
torch::jit::Stack boxArgs(Args... args) {
  // TODO Reuse stack vector instead of allocating?
  torch::jit::Stack stack;
  stack.reserve(sizeof...(Args));
  torch::jit::push(stack, std::forward<Args>(args)...);
  return stack;
}

template <class T>
inline constexpr size_t boxed_size_one() {
  static_assert(!std::is_same<std::decay_t<T>, c10::TensorOptions>::value, "need to patch this path to support TensorOptions passed by reference");
  return 1;
}

// torch::jit::push pushes 4 values for a TensorOptions; this needs to
// be kept in sync.
template <>
inline constexpr size_t boxed_size_one<c10::TensorOptions>() {
  return 4;
}

// NOTE: this could probably be simplified with C++17 fold expressions.
template <typename...>
struct BoxedSize : std::integral_constant<size_t, 0> {};
template <class T, class... Args>
struct BoxedSize<T, Args...> : std::integral_constant<size_t, boxed_size_one<T>() + BoxedSize<Args...>::value> {};

template <class... Args>
static inline constexpr size_t boxed_size() {
  return BoxedSize<Args...>::value;
}

using IValueAlignedStorage = std::aligned_storage_t<sizeof(IValue), alignof(IValue)>;

template <typename T>
C10_ALWAYS_INLINE_UNLESS_MOBILE void boxToStack(IValueAlignedStorage* dest, T& arg, int& lastIdx) {
  new (&dest[lastIdx]) IValue(arg);
  lastIdx++;
}

C10_ALWAYS_INLINE_UNLESS_MOBILE void boxToStack(IValueAlignedStorage* dest, c10::TensorOptions options, int& lastIdx) {
  new (&dest[lastIdx++]) IValue(c10::typeMetaToScalarType(options.dtype()));
  new (&dest[lastIdx++]) IValue(options.layout());
  new (&dest[lastIdx++]) IValue(options.device());
  new (&dest[lastIdx++]) IValue(options.pinned_memory());
}

inline void boxArgsToStack(IValueAlignedStorage*, int&) {}

template<typename T, typename... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE void boxArgsToStack(IValueAlignedStorage* dest, int& lastIdx, T& arg, Args &... args) {
  boxToStack(dest, arg, lastIdx);
  boxArgsToStack(dest, lastIdx, args...);
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
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    torch::jit::Stack stack = boxArgs<Args...>(std::forward<Args>(args)...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);

    if constexpr (!std::is_same_v<void, Result>) {
        // op has pushed one or more values onto the stack.
        return PopResult<Result>::call(stack);
    } else {
      // op returns void, boxed kernel has pushed nothing onto stack.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          stack.empty(),
          "Boxed kernel was expected to return no values on the stack, ",
          "but instead returned ", stack.size(), " values."
      );
    }
  }
};

//
// 3. in-place ops take a single non-const Tensor reference
// as their first argument, and return it.
//
// Note: all signatures matching this pattern are assumed to be for such ops.
// Because of this, the generated BoxedKernelWrapper specializations simply
// return the in-place argument.
//

template <class... OtherArgs>
struct BoxedKernelWrapper<
  at::Tensor&(at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  static at::Tensor& call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    at::Tensor& outArg, OtherArgs... otherArgs
  ) {
    torch::jit::Stack stack = boxArgs<at::Tensor&, OtherArgs...>(outArg, std::forward<OtherArgs>(otherArgs)...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    return outArg;
  }
};

//
// 3.5. In-process migration to make in-place ops take and return
// const references instead.
template <class... OtherArgs>
struct BoxedKernelWrapper<
  const at::Tensor&(const at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  static const at::Tensor& call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    const at::Tensor& outArg, OtherArgs... otherArgs
  ) {
    torch::jit::Stack stack = boxArgs(outArg, otherArgs...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
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
// Note: all signatures matching this pattern are assumed to be for such ops.
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
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    FirstArg firstArg, RestArgs... restArgs
  ) {
    torch::jit::Stack stack = boxArgs<FirstArg, RestArgs...>(std::forward<FirstArg>(firstArg), std::forward<RestArgs>(restArgs)...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // reusing restArgs after it has been forwarded here is ok because we know
    // that the last element is of type `Tensor&`.
    return std::get<sizeof...(RestArgs) - 1>(std::tuple<RestArgs...>{restArgs...});
  }
};

//
// 5. out of place ops that take multiple non-const Tensor references as their
// final arguments, and return them in a std::tuple.
//
// Note: all signatures matching this pattern are assumed to be for such ops.
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
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    using ArgTuple = std::tuple<Args...>;
    constexpr int RetCount = std::tuple_size<Result>();

    torch::jit::Stack stack = boxArgs<Args...>(std::forward<Args>(args)...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == RetCount,
      "Boxed kernel was expected to return ", RetCount, " values on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // reusing args after it has been forwarded here is ok because we know
    // that the last RetCount elements are of type `Tensor&`.
    auto result = guts::tuple_take<ArgTuple, -RetCount>(ArgTuple{std::forward<Args>(args)...});
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
