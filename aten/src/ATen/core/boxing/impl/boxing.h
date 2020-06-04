#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/boxing/KernelFunction.h>

#include <ATen/core/Dimname.h>

namespace c10 {
namespace impl {

//
// utils
//

// is_specialization_of
//
template <template <typename...> class T, typename U>
struct is_specialization_of: std::false_type {};

template <template <typename...> class T, typename... Us>
struct is_specialization_of<T, T<Us...>>: std::true_type {};

// is_tuple_of_lvalue_refs
//
template<class T, bool = is_specialization_of<std::tuple, T>::value>
struct is_tuple_of_lvalue_refs :
  guts::typelist::all<
    std::is_lvalue_reference, guts::typelist::from_tuple_t<T>
  >
{};

template<class T>
struct is_tuple_of_lvalue_refs<T, false> : std::false_type {};

// tuple_elements
//
template <class Tuple, size_t... ns>
constexpr auto tuple_elements(Tuple t, std::index_sequence<ns...>) {
    return std::tuple<std::tuple_element_t<ns, Tuple>...>(std::get<ns>(t)...);
}

// tuple_take
//
template <class Tuple, size_t n>
constexpr auto tuple_take(Tuple t) {
    return tuple_elements(t, std::make_index_sequence<n>{});
}

//
// boxing predicates
//

// A boxable arg type is one that IValue has a constructor for.
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

// an unboxable result is one that can be extracted from an IValue
template <typename T>
using unboxable_result =
  guts::conjunction<
    guts::disjunction<
      // using IValue(T) as a proxy for IValue.to<T>() here
      std::is_constructible<IValue, T>,
      // void returns are ok
      std::is_same<void, T>
    >,
    guts::negation<std::is_same<IntArrayRef, T>>,
    guts::negation<std::is_lvalue_reference<T>>
  >;

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
// an op signature has failed to trigger any of the partial specialiations
// that follow.
//
template<class FuncType, class Enable = void>
struct BoxedKernelWrapper {};

// 1. unsupported signatures, except inplace/outplace
// Ops whose signatures include unsupported types will generate instances
// of this class, unless their only transgression is returning one or more
// lvalue references. These are handled by special cases, below.
//
template<class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    // either not all args are boxable
    !boxable_args<Args...>::value ||
    // or the result is unboxable and not lvalue ref(s)
    (!unboxable_result<Result>::value &&
      !std::is_lvalue_reference<Result>::value &&
      !is_tuple_of_lvalue_refs<Result>::value),
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Args... args
  ) {
    // Some kernels don't need to actually box, and don't return.  If that's the
    // case, just call them anyway without a stack.  These special cases can be
    // removed once we support boxing everything.
    // See Note [named_not_supported_kernel]
    if (boxed_kernel_func == &named_not_supported_kernel) {
      named_not_supported_kernel(functor, opHandle, nullptr);  // does not return
    }

    TORCH_INTERNAL_ASSERT(
      false,
      "Unboxed call (KernelFunction::call()) on a boxed kernel with "
      "parameter or result types unsupported by BoxedKernelWrapper."
    );
  }
};

// 2. supported signatures.
//
template<class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<boxable_args<Args...>::value && unboxable_result<Result>::value, void>
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
          "Boxed kernel was expected to push exactly one return value to the stack."
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

// 3. signatures returning a single reference of the same type as
// their initial argument.
// Note that the passed kernels are assumed to be for inplace/outplace ops,
// and the generated BoxedKernelWrapper specializations will simply return
// the initial argument.
//
template<class Result, class... OtherArgs>
struct BoxedKernelWrapper<
  Result(Result, OtherArgs...),
  std::enable_if_t<
    boxable_args<Result, OtherArgs...>::value && std::is_lvalue_reference<Result>::value,
    void
  >
> {
  static Result call(
    KernelFunction::InternalBoxedKernelFunction* boxed_kernel_func,
    OperatorKernel* functor,
    const OperatorHandle& opHandle,
    Result outArg,
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

// 4. signatures returning a tuple of references, of the same type as
// the equivalent number of initial arguments.
// Note that the passed kernels are assumed to be for inplace/outplace ops,
// and the generated BoxedKernelWrapper specializations will return a tuple
// of those initial arguments.
//
template<class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    boxable_args<Args...>::value && is_tuple_of_lvalue_refs<Result>::value,
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

    using ArgTuple = std::tuple<Args...>;
    constexpr int n = std::tuple_size<Result>();
    auto result = tuple_take<ArgTuple, n>(ArgTuple{args...});
    static_assert(
        std::is_same<Result, decltype(result)>::value,
        "The parameter list of an op returning a tuple of references "
            "must begin with an equal number of parameters of the same types."
    );
    return result;
  }
};

} // impl
} // c10
