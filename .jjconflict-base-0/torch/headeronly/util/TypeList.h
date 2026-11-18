#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/TypeTraits.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace c10::guts {

template <class... T>
struct false_t : std::false_type {};
namespace typelist {

/**
 * Type holding a list of types for compile time type computations
 */
template <class... Items>
struct typelist final {
 public:
  typelist() = delete; // not for instantiation
};

/**
 * Returns the number of types in a typelist
 * Example:
 *   3  ==  size<typelist<int, int, double>>::value
 */
template <class TypeList>
struct size final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::size<T>, T must be typelist<...>.");
};
template <class... Types>
struct size<typelist<Types...>> final {
  static constexpr size_t value = sizeof...(Types);
};

/**
 * Transforms a list of types into a tuple holding these types.
 * Example:
 *   std::tuple<int, string>  ==  to_tuple_t<typelist<int, string>>
 */
template <class TypeList>
struct to_tuple final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::to_tuple<T>, T must be typelist<...>.");
};
template <class... Types>
struct to_tuple<typelist<Types...>> final {
  using type = std::tuple<Types...>;
};
template <class TypeList>
using to_tuple_t = typename to_tuple<TypeList>::type;

/**
 * Creates a typelist containing the types of a given tuple.
 * Example:
 *   typelist<int, string>  ==  from_tuple_t<std::tuple<int, string>>
 */
template <class Tuple>
struct from_tuple final {
  static_assert(
      false_t<Tuple>::value,
      "In typelist::from_tuple<T>, T must be std::tuple<...>.");
};
template <class... Types>
struct from_tuple<std::tuple<Types...>> final {
  using type = typelist<Types...>;
};
template <class Tuple>
using from_tuple_t = typename from_tuple<Tuple>::type;

/**
 * Concatenates multiple type lists.
 * Example:
 *   typelist<int, string, int>  ==  concat_t<typelist<int, string>,
 * typelist<int>>
 */
template <class... TypeLists>
struct concat final {
  static_assert(
      false_t<TypeLists...>::value,
      "In typelist::concat<T1, ...>, the T arguments each must be typelist<...>.");
};
template <class... Head1Types, class... Head2Types, class... TailLists>
struct concat<typelist<Head1Types...>, typelist<Head2Types...>, TailLists...>
    final {
  using type =
      typename concat<typelist<Head1Types..., Head2Types...>, TailLists...>::
          type;
};
template <class... HeadTypes>
struct concat<typelist<HeadTypes...>> final {
  using type = typelist<HeadTypes...>;
};
template <>
struct concat<> final {
  using type = typelist<>;
};
template <class... TypeLists>
using concat_t = typename concat<TypeLists...>::type;

/**
 * Checks if a typelist contains a certain type.
 * Examples:
 *  contains<typelist<int, string>, string> == true_type
 *  contains<typelist<int, string>, double> == false_type
 */
namespace detail {
template <class TypeList, class Type>
struct contains_impl : std::false_type {};
template <class Type, class... Types>
struct contains_impl<typelist<Types...>, Type>
    : std::bool_constant<(std::is_same_v<Types, Type> || ...)> {};
} // namespace detail
template <class TypeList, class Type>
using contains = typename detail::contains_impl<TypeList, Type>::type;

/**
 * Returns true iff the type trait is true for all types in the type list
 * Examples:
 *   true   ==  all<std::is_reference, typelist<int&, const float&&, const
 * MyClass&>>::value false  ==  all<std::is_reference, typelist<int&, const
 * float&&, MyClass>>::value
 */
template <template <class> class Condition, class TypeList>
struct all {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::all<Condition, TypeList>, the TypeList argument must be typelist<...>.");
};
template <template <class> class Condition, class... Types>
struct all<Condition, typelist<Types...>>
    : std::conjunction<Condition<Types>...> {
  static_assert(
      is_type_condition<Condition>::value,
      "In typelist::all<Condition, TypeList>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");
};

/**
 * Returns true iff the type trait is true for any type in the type list
 * Examples:
 *   true   ==  true_for_any_type<std::is_reference, typelist<int, const
 * float&&, const MyClass>>::value false  ==
 * true_for_any_type<std::is_reference, typelist<int, const float,
 * MyClass>>::value
 */
template <template <class> class Condition, class TypeList>
struct true_for_any_type final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::true_for_any_type<Condition, TypeList>, the TypeList argument must be typelist<...>.");
};
template <template <class> class Condition, class... Types>
struct true_for_any_type<Condition, typelist<Types...>> final
    : std::disjunction<Condition<Types>...> {
  static_assert(
      is_type_condition<Condition>::value,
      "In typelist::true_for_any_type<Condition, TypeList>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");
};

/**
 * Returns the first element of a type list.
 * Example:
 *   int  ==  head_t<typelist<int, string>>
 */
template <class TypeList>
struct head final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::head<T>, the T argument must be typelist<...>.");
};
template <class Head, class... Tail>
struct head<typelist<Head, Tail...>> final {
  using type = Head;
};
template <class TypeList>
using head_t = typename head<TypeList>::type;

/**
 * Returns the first element of a type list, or the specified default if the
 * type list is empty. Example: int  ==  head_t<bool, typelist<int, string>>
 *   bool  ==  head_t<bool, typelist<>>
 */
template <class Default, class TypeList>
struct head_with_default final {
  using type = Default;
};
template <class Default, class Head, class... Tail>
struct head_with_default<Default, typelist<Head, Tail...>> final {
  using type = Head;
};
template <class Default, class TypeList>
using head_with_default_t = typename head_with_default<Default, TypeList>::type;

/**
 * Returns the N-th element of a type list.
 * Example:
 * int == element_t<1, typelist<float, int, char>>
 */

/// Base template.
template <size_t Index, class TypeList>
struct element final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::element<T>, the T argument must be typelist<...>.");
};

/// Successful case, we have reached the zero index and can "return" the head
/// type.
template <class Head, class... Tail>
struct element<0, typelist<Head, Tail...>> {
  using type = Head;
};

/// Error case, we have an index but ran out of types! It will only be selected
/// if `Ts...` is actually empty!
template <size_t Index, class... Ts>
struct element<Index, typelist<Ts...>> {
  static_assert(
      Index < sizeof...(Ts),
      "Index is out of bounds in typelist::element");
};

/// Shave off types until we hit the <0, Head, Tail...> or <Index> case.
template <size_t Index, class Head, class... Tail>
struct element<Index, typelist<Head, Tail...>>
    : element<Index - 1, typelist<Tail...>> {};

/// Convenience alias.
template <size_t Index, class TypeList>
using element_t = typename element<Index, TypeList>::type;

/**
 * Take/drop a number of arguments from a typelist.
 * Example:
 *   typelist<int, string> == take_t<typelist<int, string, bool>, 2>
 *   typelist<bool> == drop_t<typelist<int, string, bool>, 2>
 */
namespace detail {
template <class TypeList, size_t offset, class IndexSequence>
struct take_elements final {};

template <class TypeList, size_t offset, size_t... Indices>
struct take_elements<TypeList, offset, std::index_sequence<Indices...>> final {
  using type = typelist<typename element<offset + Indices, TypeList>::type...>;
};
} // namespace detail

template <class TypeList, size_t num>
struct take final {
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::take<T, num>, the T argument must be typelist<...>.");
  static_assert(
      num <= size<TypeList>::value,
      "Tried to typelist::take more elements than there are in the list");
  using type = typename detail::
      take_elements<TypeList, 0, std::make_index_sequence<num>>::type;
};
template <class TypeList, size_t num>
using take_t = typename take<TypeList, num>::type;

template <class TypeList, size_t num>
struct drop final {
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  static_assert(
      num <= size<TypeList>::value,
      "Tried to typelist::drop more elements than there are in the list");
  using type = typename detail::take_elements<
      TypeList,
      num,
      std::make_index_sequence<size<TypeList>::value - num>>::type;
};
template <class TypeList, size_t num>
using drop_t = typename drop<TypeList, num>::type;

/**
 * Like drop, but returns an empty list rather than an assertion error if `num`
 * is larger than the size of the TypeList.
 * Example:
 *   typelist<> == drop_if_nonempty_t<typelist<string, bool>, 2>
 *   typelist<> == drop_if_nonempty_t<typelist<int, string, bool>, 3>
 */
template <class TypeList, size_t num>
struct drop_if_nonempty final {
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  using type = typename detail::take_elements<
      TypeList,
      std::min(num, size<TypeList>::value),
      std::make_index_sequence<
          size<TypeList>::value - std::min(num, size<TypeList>::value)>>::type;
};
template <class TypeList, size_t num>
using drop_if_nonempty_t = typename drop_if_nonempty<TypeList, num>::type;

} // namespace typelist
} // namespace c10::guts

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

namespace guts {
using c10::guts::false_t;

namespace typelist {
using c10::guts::typelist::all;
using c10::guts::typelist::concat_t;
using c10::guts::typelist::contains;
using c10::guts::typelist::drop_if_nonempty_t;
using c10::guts::typelist::drop_t;
using c10::guts::typelist::from_tuple_t;
using c10::guts::typelist::head_t;
using c10::guts::typelist::head_with_default_t;
using c10::guts::typelist::size;
using c10::guts::typelist::take_t;
using c10::guts::typelist::to_tuple_t;
using c10::guts::typelist::true_for_any_type;
using c10::guts::typelist::typelist;

} // namespace typelist

} // namespace guts

HIDDEN_NAMESPACE_END(torch, headeronly)
