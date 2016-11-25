#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace thd {
/*
 * The following notation comes from:
 * docs.python.org/3.5/library/struct.html#module-struct
 * except from 'T', which stands for Tensor
 */

enum class TensorType : char {
  CHAR = 'c',
  UCHAR = 'B',
  FLOAT = 'f',
  DOUBLE = 'd',
  SHORT = 'h',
  USHORT = 'H',
  INT = 'i',
  UINT = 'I',
  LONG = 'l',
  ULONG = 'L',
  LONG_LONG = 'q',
  ULONG_LONG = 'Q',
  TENSOR = 'T',
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
} // namespace thd
