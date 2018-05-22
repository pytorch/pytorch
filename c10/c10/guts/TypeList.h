#pragma once

#include "C++17.h"

namespace c10 { namespace guts { namespace typelist {

/**
 * Type holding a list of types for compile time type computations
 */
template<class... Items> struct typelist final {
  /**
   * Transforms a list of types into a tuple holding these types
   */
  using tuple_type = std::tuple<Items...>;

  /**
   * Number of types in the list
   */
  static constexpr size_t size = sizeof...(Items);
};



/**
 * Creates a typelist containing the types of a given tuple.
 * Example:
 *   typelist<int, string>  ==  from_tuple_t<std::tuple<int, string>>
 */
template<class Tuple> struct from_tuple;
template<class... Types> struct from_tuple<std::tuple<Types...>> final {
  using type = typelist<Types...>;
};
template<class Tuple> using from_tuple_t = typename from_tuple<Tuple>::type;



/**
 * Concatenates multiple type lists.
 * Example:
 *   typelist<int, string, int>  ==  concat_t<typelist<int, string>, typelist<int>>
 */
template<class... TypeLists> struct concat;
template<class... Head1Types, class... Head2Types, class... TailLists>
struct concat<typelist<Head1Types...>, typelist<Head2Types...>, TailLists...> final {
  using type = typename concat<typelist<Head1Types..., Head2Types...>, TailLists...>::type;
};
template<class... HeadTypes>
struct concat<typelist<HeadTypes...>> final {
  using type = typelist<HeadTypes...>;
};
template<>
struct concat<> final {
  using type = typelist<>;
};
template<class... TypeLists> using concat_t = typename concat<TypeLists...>::type;



/**
 * Filters the types in a type list by a type trait.
 * Examples:
 *   typelist<int&, const string&&>  ==  filter_t<std::is_reference, typelist<void, string, int&, bool, const string&&, int>>
 */
template<template <class> class Condition, class TypeList> struct filter;
template<template <class> class Condition, class Head, class... Tail>
struct filter<Condition, typelist<Head, Tail...>> final {
  using type = std::conditional_t<
    Condition<Head>::value,
    concat_t<typelist<Head>, typename filter<Condition, typelist<Tail...>>::type>,
    typename filter<Condition, typelist<Tail...>>::type
  >;
};
template<template <class> class Condition>
struct filter<Condition, typelist<>> final {
  using type = typelist<>;
};
template<template <class> class Condition, class TypeList>
using filter_t = typename filter<Condition, TypeList>::type;



/**
 * Counts how many types in the list fulfill a type trait
 * Examples:
 *   2  ==  count_if<std::is_reference, typelist<void, string, int&, bool, const string&&, int>>
 */
template<template <class> class Condition, class TypeList>
struct count_if final {
  // TODO Direct implementation might be faster
  static constexpr size_t value = filter_t<Condition, TypeList>::size;
};



/**
 * Returns true iff the type trait is true for all types in the type list
 * Examples:
 *   true   ==  true_for_each_type<std::is_reference, typelist<int&, const float&&, const MyClass&>>::value
 *   false  ==  true_for_each_type<std::is_reference, typelist<int&, const float&&, MyClass>>::value
 */
template<template <class> class Condition, class TypeList> struct true_for_each_type;
template<template <class> class Condition, class... Types>
struct true_for_each_type<Condition, typelist<Types...>> final
: guts::conjunction<Condition<Types>...> {};



/**
 * Maps types of a type list using a type trait
 * Example:
 *  typelist<int&, double&, string&>  ==  map_t<std::add_lvalue_reference_t, typelist<int, double, string>>
 */
template<template <class> class Mapper, class TypeList> struct map;
template<template <class> class Mapper, class... Types>
struct map<Mapper, typelist<Types...>> final {
  using type = typelist<Mapper<Types>...>;
};
template<template <class> class Mapper, class TypeList>
using map_t = typename map<Mapper, TypeList>::type;



/**
 * Returns the first element of a type list.
 * Example:
 *   int  ==  head_t<typelist<int, string>>
 */
template<class TypeList> struct head;
template<class Head, class... Tail> struct head<typelist<Head, Tail...>> final {
  using type = Head;
};
template<class TypeList> using head_t = typename head<TypeList>::type;



/**
 * Reverses a typelist.
 * Example:
 *   typelist<int, string>  == reverse_t<typelist<string, int>>
 */
template<class TypeList> struct reverse;
template<class Head, class... Tail> struct reverse<typelist<Head, Tail...>> final {
  using type = concat_t<typename reverse<typelist<Tail...>>::type, typelist<Head>>;
};
template<> struct reverse<typelist<>> final {
  using type = typelist<>;
};
template<class TypeList> using reverse_t = typename reverse<TypeList>::type;



/**
 * Maps a list of types into a list of values.
 * Examples:
 *   auto sizes =
 *     map_types_to_values<typelist<int64_t, bool, uint32_t>>(
 *       [] (auto t) { return sizeof(decltype(t)::type); }
 *     );
 *   //  sizes  ==  std::tuple<size_t, size_t, size_t>{8, 1, 4}
 *
 *   auto shared_ptrs =
 *     map_types_to_values<typelist<int, double>>(
 *       [] (auto t) { return make_shared<typename decltype(t)::type>(); }
 *     );
 *   // shared_ptrs == std::tuple<shared_ptr<int>, shared_ptr<double>>()
 */
namespace details {
template<class T> struct type_ final {
    using type = T;
};
template<class TypeList> struct map_types_to_values;
template<class... Types> struct map_types_to_values<typelist<Types...>> final {
  template<class Func>
  static std::tuple<std::result_of_t<Func(type_<Types>)>...> call(Func&& func) {
    return { std::forward<Func>(func)(type_<Types>())... };
  }
};
}

template<class TypeList, class Func> auto map_types_to_values(Func&& func) {
  return details::map_types_to_values<TypeList>::call(std::forward<Func>(func));
}


}}}
