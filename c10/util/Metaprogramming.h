#pragma once

#include <c10/util/Array.h>
#include <c10/util/TypeList.h>
#include <array>
#include <functional>
#include <type_traits>

namespace c10 {
namespace guts {

/**
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::parameter_types::tuple_type
 * // A == tuple<float, double>
 */
template <class Func>
struct function_traits {
  static_assert(
      !std::is_same<Func, Func>::value,
      "In function_traits<Func>, Func must be a plain function type.");
};
template <class Result, class... Args>
struct function_traits<Result(Args...)> {
  using func_type = Result(Args...);
  using return_type = Result;
  using parameter_types = typelist::typelist<Args...>;
  static constexpr auto number_of_parameters = sizeof...(Args);
};

/**
 * infer_function_traits: creates a `function_traits` type for a simple
 * function (pointer) or functor (lambda/struct). Currently does not support
 * class methods.
 */

template <typename Functor>
struct infer_function_traits {
  using type = function_traits<
      c10::guts::detail::strip_class_t<decltype(&Functor::operator())>>;
};

template <typename Result, typename... Args>
struct infer_function_traits<Result (*)(Args...)> {
  using type = function_traits<Result(Args...)>;
};

template <typename Result, typename... Args>
struct infer_function_traits<Result(Args...)> {
  using type = function_traits<Result(Args...)>;
};

template <typename T>
using infer_function_traits_t = typename infer_function_traits<T>::type;

/**
 * make_function_traits: creates a `function_traits` type given a Return type
 * and a typelist of Argument types
 *
 * Example:
 * bool f(int, int);
 *
 * infer_function_traits_t<f> == make_function_traits_t<bool,
 * typelist::typelist<int, int>>
 */
template <typename Result, typename ArgList>
struct make_function_traits {
  static_assert(
      false_t<ArgList>::value,
      "In guts::make_function_traits<Result, TypeList>, the ArgList argument must be typelist<...>.");
};

template <typename Result, typename... Args>
struct make_function_traits<Result, typelist::typelist<Args...>> {
  using type = function_traits<Result(Args...)>;
};

template <typename Result, typename ArgList>
using make_function_traits_t =
    typename make_function_traits<Result, ArgList>::type;

/**
 * Use extract_arg_by_filtered_index to return the i-th argument whose
 * type fulfills a given type trait. The argument itself is perfectly forwarded.
 *
 * Example:
 * std::string arg1 = "Hello";
 * std::string arg2 = "World";
 * std::string&& result = extract_arg_by_filtered_index<is_string, 1>(0,
 * arg1, 2.0, std::move(arg2));
 *
 * Warning: Taking the result by rvalue reference can cause segfaults because
 * ownership will not be passed on from the original reference. The original
 * reference dies after the expression and the resulting
 */
namespace detail {
template <
    template <class>
    class Condition,
    size_t index,
    class Enable,
    class... Args>
struct extract_arg_by_filtered_index_;
template <
    template <class>
    class Condition,
    size_t index,
    class Head,
    class... Tail>
struct extract_arg_by_filtered_index_<
    Condition,
    index,
    std::enable_if_t<!Condition<Head>::value>,
    Head,
    Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index, void, Tail...>::
        call(std::forward<Tail>(tail)...);
  }
};
template <
    template <class>
    class Condition,
    size_t index,
    class Head,
    class... Tail>
struct extract_arg_by_filtered_index_<
    Condition,
    index,
    std::enable_if_t<Condition<Head>::value && index != 0>,
    Head,
    Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index - 1, void, Tail...>::
        call(std::forward<Tail>(tail)...);
  }
};
template <template <class> class Condition, size_t index>
struct extract_arg_by_filtered_index_<Condition, index, void> {
  static void call() {
    static_assert(
        index != index, "extract_arg_by_filtered_index out of range.");
  }
};
template <
    template <class>
    class Condition,
    size_t index,
    class Head,
    class... Tail>
struct extract_arg_by_filtered_index_<
    Condition,
    index,
    std::enable_if_t<Condition<Head>::value && index == 0>,
    Head,
    Tail...> {
  static decltype(auto) call(Head&& head, Tail&&... /*tail*/) {
    return std::forward<Head>(head);
  }
};
} // namespace detail
template <template <class> class Condition, size_t index, class... Args>
decltype(auto) extract_arg_by_filtered_index(Args&&... args) {
  static_assert(
      is_type_condition<Condition>::value,
      "In extract_arg_by_filtered_index, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");
  return detail::
      extract_arg_by_filtered_index_<Condition, index, void, Args...>::call(
          std::forward<Args>(args)...);
}

/**
 * Use filter_map to map a subset of the arguments to values.
 * The subset is defined by type traits, and will be evaluated at compile time.
 * At runtime, it will just loop over the pre-filtered arguments to create an
 * std::array.
 *
 * Example:
 *  std::array<double, 2> result = filter_map<double, std::is_integral>([] (auto
 * a) {return (double)a;}, 3, "bla", 4);
 *  // result == {3.0, 4.0}
 */
namespace detail {

template <class ResultType, size_t num_results>
struct filter_map_ {
  template <
      template <class>
      class Condition,
      class Mapper,
      class... Args,
      size_t... INDEX>
  static guts::array<ResultType, num_results> call(
      const Mapper& mapper,
      std::index_sequence<INDEX...>,
      Args&&... args) {
    return guts::array<ResultType, num_results>{
        mapper(extract_arg_by_filtered_index<Condition, INDEX>(
            std::forward<Args>(args)...))...};
  }
};
template <class ResultType>
struct filter_map_<ResultType, 0> {
  template <
      template <class>
      class Condition,
      class Mapper,
      class... Args,
      size_t... INDEX>
  static guts::array<ResultType, 0> call(
      const Mapper& /*mapper*/,
      std::index_sequence<INDEX...>,
      Args&&... /*args*/) {
    return guts::array<ResultType, 0>{};
  }
};
} // namespace detail

template <
    class ResultType,
    template <class>
    class Condition,
    class Mapper,
    class... Args>
decltype(auto) filter_map(const Mapper& mapper, Args&&... args) {
  static_assert(
      is_type_condition<Condition>::value,
      "In filter_map<Result, Condition>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");

  static constexpr size_t num_results =
      typelist::count_if<Condition, typelist::typelist<Args...>>::value;
  return detail::filter_map_<ResultType, num_results>::
      template call<Condition, Mapper, Args...>(
          mapper,
          std::make_index_sequence<num_results>(),
          std::forward<Args>(args)...);
}

/**
 * make_offset_index_sequence<Start, N>
 * Like make_index_sequence<N>, but starting from Start instead of 0.
 *
 * Example:
 *  make_offset_index_sequence<10, 3> == std::index_sequence<10, 11, 12>
 */
template <size_t Start, size_t N, size_t... Is>
struct make_offset_index_sequence_impl
    : make_offset_index_sequence_impl<Start, N - 1, Start + N - 1, Is...> {
  static_assert(
      static_cast<int>(Start) >= 0,
      "make_offset_index_sequence: Start < 0");
  static_assert(static_cast<int>(N) >= 0, "make_offset_index_sequence: N < 0");
};

template <size_t Start, size_t... Is>
struct make_offset_index_sequence_impl<Start, 0, Is...> {
  typedef std::index_sequence<Is...> type;
};

template <size_t Start, size_t N>
using make_offset_index_sequence =
    typename make_offset_index_sequence_impl<Start, N>::type;

/**
 * Use tuple_elements to extract a position-indexed subset of elements
 * from the argument tuple into a result tuple.
 *
 * Example:
 *  std::tuple<int, const char*, double> t = std::make_tuple(0, "HEY", 2.0);
 *  std::tuple<int, double> result = tuple_elements(t, std::index_sequence<0,
 * 2>());
 */
template <class Tuple, size_t... Is>
constexpr auto tuple_elements(Tuple t, std::index_sequence<Is...>) {
  return std::tuple<std::tuple_element_t<Is, Tuple>...>(std::get<Is>(t)...);
}

/**
 * Use tuple_take to extract the first or last n elements from the argument
 * tuple into a result tuple.
 *
 * Example:
 *  std::tuple<int, const char*, double> t = std::make_tuple(0, "HEY", 2.0);
 *  std::tuple<int, const char*> first_two = tuple_take<decltype(t), 2>(t);
 *  std::tuple<const char*, double> last_two = tuple_take<decltype(t), -2>(t);
 */
template <class Tuple, int N, class Enable = void>
struct TupleTake {};

template <class Tuple, int N>
struct TupleTake<Tuple, N, std::enable_if_t<N >= 0, void>> {
  static auto call(Tuple t) {
    constexpr size_t size = std::tuple_size<Tuple>();
    static_assert(N <= size, "tuple_take: N > size");
    return tuple_elements(t, std::make_index_sequence<N>{});
  }
};

template <class Tuple, int N>
    struct TupleTake < Tuple,
    N, std::enable_if_t<N<0, void>> {
  static auto call(Tuple t) {
    constexpr size_t size = std::tuple_size<Tuple>();
    static_assert(-N <= size, "tuple_take: -N > size");
    return tuple_elements(t, make_offset_index_sequence<size + N, -N>{});
  }
};

template <class Tuple, int N>
auto tuple_take(Tuple t) {
  return TupleTake<Tuple, N>::call(t);
}

/**
 * Use tuple_slice to extract a contiguous subtuple from the argument.
 *
 * Example:
 *  std::tuple<int, const char*, double, bool> t = std::make_tuple(0,
 * "HEY", 2.0, false); std::tuple<int, const char*> middle_two =
 * tuple_slice<decltype(t), 1, 2>(t);
 */
template <class Tuple, size_t Start, size_t N>
constexpr auto tuple_slice(Tuple t) {
  constexpr size_t size = std::tuple_size<Tuple>();
  static_assert(Start + N <= size, "tuple_slice: Start + N > size");
  return tuple_elements(t, make_offset_index_sequence<Start, N>{});
}

/**
 * Use tuple_map to run a mapping function over a tuple to get a new tuple.
 *
 * Example 1:
 *   auto result = tuple_map(std::tuple<int32_t, int32_t, int32_t>(3, 4, 5), []
 * (int32_t a) -> int16_t {return a+1;});
 *   // result == std::tuple<int16_t, int16_t, int16_t>(4, 5, 6)
 *
 * Example 2:
 *   struct Mapper {
 *     std::string operator()(int32_t a) const {
 *       return std::to_string(a);
 *     }
 *     int64_t operator()(const std::string& a) const {
 *        return atoi(a.c_str());
 *     }
 *   };
 *   auto result = tuple_map(std::tuple<int32_t, std::string>(3, "4"),
 * Mapper());
 *   // result == std::tuple<std::string, int64_t>("3", 4)
 *
 * Example 3:
 *   struct A final {
 *    int32_t func() {
 *      return 5;
 *    }
 *  };
 *  struct B final {
 *    std::string func() {
 *      return "5";
 *    }
 *  };
 *  auto result = tuple_map(std::make_tuple(A(), B()), [] (auto a) { return
 * a.func(); });
 *  // result == std::tuple<int32_t, std::string>(5, "5");
 */
namespace detail {
template <class Mapper, class... Args, size_t... Indices>
auto tuple_map(
    std::tuple<Args...>&& tuple,
    const Mapper& mapper,
    std::index_sequence<Indices...>) {
  return std::tuple<decltype(mapper(std::forward<Args>(std::get<Indices>(
      tuple))))...>(mapper(std::forward<Args>(std::get<Indices>(tuple)))...);
}
} // namespace detail

template <class Mapper, class... Args>
auto tuple_map(std::tuple<Args...>&& tuple, const Mapper& mapper) {
  return detail::tuple_map(
      std::move(tuple), mapper, std::index_sequence_for<Args...>());
}

/**
 * tuple_concat concatenates several tuples into one.
 */

namespace detail {
// extract_tuple_element_by_index is a helper that takes a list of tuples and
// extracts the i-th element in a flattened view of the tuples. Example:
// extract_tuple_element_by_index<3>(tuple(2,3), tuple(4,5), tuple(6,7)) == 5.

template <
    size_t index,
    class HeadTuple,
    class... TailTuples,
    std::enable_if_t<
        index<std::tuple_size<HeadTuple>::value, int> = 0> decltype(auto)
        extract_tuple_element_by_index(
            HeadTuple&& head_tuple,
            TailTuples&&... /*tail_tuples*/) {
  // TODO if constexpr instead of enable_if
  return std::get<index>(std::forward<HeadTuple>(head_tuple));
}

template <
    size_t index,
    class HeadTuple,
    class... TailTuples,
    std::enable_if_t<index >= std::tuple_size<HeadTuple>::value, int> = 0>
decltype(auto) extract_tuple_element_by_index(
    HeadTuple&& /*head_tuple*/,
    TailTuples&&... tail_tuples) {
  // TODO if constexpr instead of enable_if
  return extract_tuple_element_by_index<
      index - std::tuple_size<HeadTuple>::value,
      TailTuples...>(std::forward<TailTuples>(tail_tuples)...);
}

static_assert(
    std::is_same<
        int&&,
        decltype(extract_tuple_element_by_index<2>(
            std::tuple<int32_t>(2),
            std::tuple<int32_t&&, int32_t>(std::declval<int32_t>(), 3)))>::
        value,
    "extract_tuple_element_by_index should return rvalue references if the tuple contains them. It should not move them into a value");

template <class ConcatenatedTuple, class... Tuples, size_t... ElementIndices>
auto tuple_concat(Tuples&&... tuples, std::index_sequence<ElementIndices...>) {
  return ConcatenatedTuple(extract_tuple_element_by_index<ElementIndices>(
      std::forward<Tuples>(tuples)...)...);
}
} // namespace detail

template <class... Tuples>
auto tuple_concat(Tuples&&... tuples) {
  using flattened_types =
      guts::typelist::concat_t<guts::typelist::from_tuple_t<Tuples>...>;
  using concatenated_tuple = guts::typelist::to_tuple_t<flattened_types>;
  constexpr size_t num_elements = guts::typelist::size<flattened_types>::value;
  return detail::tuple_concat<concatenated_tuple, Tuples...>(
      std::forward<Tuples>(tuples)...,
      std::make_index_sequence<num_elements>());
}

/**
 * Concatenate multiple integer sequences
 * Example:
 *   concat_iseq_t<std::index_sequence<2, 5, 3>, std::index_sequence<4, 2>,
 * std::index_sequence<5>>
 *     == std::index_sequence<2, 5, 3, 4, 2, 5>
 */
template <class... ISeqs>
struct concat_iseq {
  static_assert(
      false_t<ISeqs...>::value,
      "In concat_iseq<T1, ...>, the T arguments each must be std::integer_sequence<...> with the same IntType.");
};
template <>
struct concat_iseq<> {
  using type = std::index_sequence<>;
};
template <class IntType, IntType... Indices>
struct concat_iseq<std::integer_sequence<IntType, Indices...>> {
  using type = std::integer_sequence<IntType, Indices...>;
};
template <
    class IntType,
    IntType... Head1Indices,
    IntType... Head2Indices,
    class... TailISeqs>
struct concat_iseq<
    std::integer_sequence<IntType, Head1Indices...>,
    std::integer_sequence<IntType, Head2Indices...>,
    TailISeqs...> {
  using type = typename concat_iseq<
      std::integer_sequence<IntType, Head1Indices..., Head2Indices...>,
      TailISeqs...>::type;
};
template <class... ISeqs>
using concat_iseq_t = typename concat_iseq<ISeqs...>::type;

} // namespace guts
} // namespace c10
