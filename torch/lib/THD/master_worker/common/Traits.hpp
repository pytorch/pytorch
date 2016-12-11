#include <type_traits>
#include <tuple>

#include "master_worker/master/THDTensor.h"
#include "master_worker/master/THDStorage.h"

namespace thd {

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

using THDTensorTypes = std::tuple<
    THDByteTensor,
    THDCharTensor,
    THDShortTensor,
    THDIntTensor,
    THDLongTensor,
    THDFloatTensor,
    THDDoubleTensor
>;

using THDStorageTypes = std::tuple<
    THDByteStorage,
    THDCharStorage,
    THDShortStorage,
    THDIntStorage,
    THDLongStorage,
    THDFloatStorage,
    THDDoubleStorage
>;

template<template<typename> class Trait, typename U>
struct map_to_ptr {};

template<template <typename> class Trait, typename... Types>
struct map_to_ptr<Trait, std::tuple<Types...>> {
  using type = std::tuple<
    typename std::add_pointer<Types>::type...,
    typename std::add_pointer<typename Trait<Types>::type>::type...
  >;
};

using THDTensorPtrTypes = map_to_ptr<std::add_const, THDTensorTypes>::type;
using THDStoragePtrTypes = map_to_ptr<std::add_const, THDStorageTypes>::type;

} // namespace thd
