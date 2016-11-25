#pragma once

#include "master_worker/master/THDTensor.h"
#include "TensorTypeTraits.hpp"

#include <type_traits>
#include <tuple>

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

template <typename T>
struct tensor_type_char {};

template <>
struct tensor_type_char<THDByteTensor>{
  static constexpr char value = static_cast<char>(TensorType::UCHAR);
};

template <>
struct tensor_type_char<THDCharTensor>{
  static constexpr char value = static_cast<char>(TensorType::CHAR);
};

template <>
struct tensor_type_char<THDShortTensor>{
  static constexpr char value = static_cast<char>(TensorType::SHORT);
};

template <>
struct tensor_type_char<THDIntTensor>{
  static constexpr char value = static_cast<char>(TensorType::INT);
};

template <>
struct tensor_type_char<THDLongTensor>{
  static constexpr char value = static_cast<char>(TensorType::LONG);
};

template <>
struct tensor_type_char<THDFloatTensor>{
  static constexpr char value = static_cast<char>(TensorType::FLOAT);
};

template <>
struct tensor_type_char<THDDoubleTensor>{
  static constexpr char value = static_cast<char>(TensorType::DOUBLE);
};

} // namespace thd
