#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "Storage.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace thd {

template<typename real>
struct interface_traits {
  using scalar_type = typename std::conditional<
    std::is_floating_point<real>::value,
    double,
    long long>::type;
  using tensor_interface_type = TensorScalarInterface<scalar_type>;
  using storage_interface_type = StorageScalarInterface<scalar_type>;
};

template<typename T>
struct type_traits {};

template<>
struct type_traits<char> {
  static constexpr Type type = Type::CHAR;
};

template<>
struct type_traits<unsigned char> {
  static constexpr Type type = Type::UCHAR;
};

template<>
struct type_traits<float> {
  static constexpr Type type = Type::FLOAT;
};

template<>
struct type_traits<double> {
  static constexpr Type type = Type::DOUBLE;
};

template<>
struct type_traits<short> {
  static constexpr Type type = Type::SHORT;
};

template<>
struct type_traits<unsigned short> {
  static constexpr Type type = Type::USHORT;
};

template<>
struct type_traits<int> {
  static constexpr Type type = Type::INT;
};

template<>
struct type_traits<unsigned int> {
  static constexpr Type type = Type::UINT;
};

template<>
struct type_traits<long> {
  static constexpr Type type = Type::LONG;
};

template<>
struct type_traits<unsigned long> {
  static constexpr Type type = Type::ULONG;
};

template<>
struct type_traits<long long> {
  static constexpr Type type = Type::LONG_LONG;
};

template<>
struct type_traits<unsigned long long> {
  static constexpr Type type = Type::ULONG_LONG;
};


} // namespace thd
