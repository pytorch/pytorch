#pragma once

#include "master_worker/master/THDTensor.h"

#include <type_traits>
#include <tuple>

template<typename...>
struct or_trait : std::false_type {};

template<typename T>
struct or_trait<T> : T {};

template <typename T, typename... Ts>
struct or_trait<T, Ts...>
  : std::conditional<T::value, T, or_trait<Ts...>>::type {};

template <typename T, typename U>
struct is_any_of : std::false_type {};

template <typename T, typename U>
struct is_any_of<T, std::tuple<U>> : std::is_same<T, U> {};

template <typename T, typename Head, typename... Tail>
struct is_any_of<T, std::tuple<Head, Tail...>>
  : or_trait<std::is_same<T, Head>, is_any_of<T, std::tuple<Tail...>>> {};

using THDTensorTypes = std::tuple<THDByteTensor>;
