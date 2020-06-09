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
  guts::typelist::all<std::is_lvalue_reference, guts::typelist::from_tuple_t<T>>
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
using can_box =
  guts::disjunction<
    std::is_constructible<IValue, T>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, T>
  >;

template <typename... Ts>
using can_box_all = guts::conjunction<can_box<std::decay_t<Ts>>...>;

// an unboxable result is one that can be extracted from an IValue
template <typename T>
using can_unbox =
  guts::conjunction<
    guts::disjunction<
      // using IValue(T) as a proxy for IValue.to<T>() here
      std::is_constructible<IValue, T>,
      // void returns are ok
      std::is_same<void, T>
    >,
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

// 1. Unsupported type traps.
//
// These specializations capture the remaining gaps in boxing support.
// Rather than triggering compile errors, we generate boxed kernels that
// raise runtime errors. As support for these types is added, the
// specializations can be removed.
//

// at::Dimname
template <class... Args>
using has_dimname_arg =
  guts::disjunction<
    std::is_same<at::Dimname, std::decay_t<Args>>...,
    std::is_same<c10::ArrayRef<at::Dimname>, std::decay_t<Args>>...,
    std::is_same<c10::optional<c10::ArrayRef<at::Dimname>>, std::decay_t<Args>>...
  >;

template<class Result, class... Args>
struct BoxedKernelWrapper<Result(Args...), std::enable_if_t<has_dimname_arg<Args...>::value, void>> {
  static Result call(KernelFunction::InternalBoxedKernelFunction*, OperatorKernel*, const OperatorHandle&, Args... args) {
    TORCH_INTERNAL_ASSERT(false, "Call to a boxed kernel with unboxable parameter type at::Dimname.");
  }
};

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
  std::enable_if_t<can_box_all<Args...>::value && can_unbox<Result>::value, void>
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
          "Boxed kernel for op with void return type pushed one or more return values to the stack."
        );
      }
    );
  }
};

// 3. signatures returning a single reference of the same type as
// their initial argument.
//
// Note that the passed kernels are assumed to be for inplace/outplace ops,
// and the generated BoxedKernelWrapper specializations will simply return
// the initial argument.
//
template<class Result, class... OtherArgs>
struct BoxedKernelWrapper<
  Result(Result, OtherArgs...),
  std::enable_if_t<
    can_box_all<Result, OtherArgs...>::value && std::is_lvalue_reference<Result>::value,
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
    can_box_all<Args...>::value && is_tuple_of_lvalue_refs<Result>::value,
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
