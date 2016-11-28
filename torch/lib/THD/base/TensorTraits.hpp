#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "master_worker/master/THDTensor.h"
#include "Tensor.hpp"
#include "Type.hpp"

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
  static constexpr Type type = Type::CHAR;
};

template<>
struct tensor_type_traits<unsigned char> {
  static constexpr Type type = Type::UCHAR;
};

template<>
struct tensor_type_traits<float> {
  static constexpr Type type = Type::FLOAT;
};

template<>
struct tensor_type_traits<double> {
  static constexpr Type type = Type::DOUBLE;
};

template<>
struct tensor_type_traits<short> {
  static constexpr Type type = Type::SHORT;
};

template<>
struct tensor_type_traits<unsigned short> {
  static constexpr Type type = Type::USHORT;
};

template<>
struct tensor_type_traits<int> {
  static constexpr Type type = Type::INT;
};

template<>
struct tensor_type_traits<unsigned int> {
  static constexpr Type type = Type::UINT;
};

template<>
struct tensor_type_traits<long> {
  static constexpr Type type = Type::LONG;
};

template<>
struct tensor_type_traits<unsigned long> {
  static constexpr Type type = Type::ULONG;
};

template<>
struct tensor_type_traits<long long> {
  static constexpr Type type = Type::LONG_LONG;
};

template<>
struct tensor_type_traits<unsigned long long> {
  static constexpr Type type = Type::ULONG_LONG;
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

template<typename T>
struct map_to_ptr {};

template<typename... Types>
struct map_to_ptr<std::tuple<Types...>> {
  using type = std::tuple<typename std::add_pointer<Types>::type...>;
};

using THDTensorPtrTypes = map_to_ptr<THDTensorTypes>::type;

} // namespace thd
