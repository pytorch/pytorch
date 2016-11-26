#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "master_worker/master/THDTensor.h"
#include "Tensor.hpp"
#include "TensorType.hpp"

namespace thd {

template<typename real>
struct tensor_interface_traits {
  using scalar_type = typename std::conditional<
    std::is_floating_point<real>::value,
    double,
    long long>::type;
  using interface_type = TensorScalarInterface<scalar_type>;
};

template<typename T>
struct tensor_type_traits {};

template<>
struct tensor_type_traits<char> {
  static constexpr TensorType type = TensorType::CHAR;
};

template<>
struct tensor_type_traits<unsigned char> {
  static constexpr TensorType type = TensorType::UCHAR;
};

template<>
struct tensor_type_traits<float> {
  static constexpr TensorType type = TensorType::FLOAT;
};

template<>
struct tensor_type_traits<double> {
  static constexpr TensorType type = TensorType::DOUBLE;
};

template<>
struct tensor_type_traits<short> {
  static constexpr TensorType type = TensorType::SHORT;
};

template<>
struct tensor_type_traits<unsigned short> {
  static constexpr TensorType type = TensorType::USHORT;
};

template<>
struct tensor_type_traits<int> {
  static constexpr TensorType type = TensorType::INT;
};

template<>
struct tensor_type_traits<unsigned int> {
  static constexpr TensorType type = TensorType::UINT;
};

template<>
struct tensor_type_traits<long> {
  static constexpr TensorType type = TensorType::LONG;
};

template<>
struct tensor_type_traits<unsigned long> {
  static constexpr TensorType type = TensorType::ULONG;
};

template<>
struct tensor_type_traits<long long> {
  static constexpr TensorType type = TensorType::LONG_LONG;
};

template<>
struct tensor_type_traits<unsigned long long> {
  static constexpr TensorType type = TensorType::ULONG_LONG;
};

static const std::unordered_map<char, TensorType> format_to_type = {
  {'c', TensorType::CHAR},
  {'f', TensorType::FLOAT},
  {'d', TensorType::DOUBLE},
  {'h', TensorType::SHORT},
  {'H', TensorType::USHORT},
  {'i', TensorType::INT},
  {'I', TensorType::UINT},
  {'l', TensorType::LONG},
  {'L', TensorType::ULONG},
  {'q', TensorType::LONG_LONG},
  {'Q', TensorType::ULONG_LONG},
};

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
