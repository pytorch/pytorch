#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace thd {

/*
 * The following notation comes from:
 * docs.python.org/3.5/library/struct.html#module-struct
 * except from 'T', which stands for Tensor
 */

enum class RPCType : char {
  CHAR = 'c',
  UCHAR = 'B',
  FLOAT = 'f',
  DOUBLE = 'd',
  HALF = 'a',
  SHORT = 'h',
  USHORT = 'H',
  INT = 'i',
  UINT = 'I',
  LONG = 'l',
  ULONG = 'L',
  LONG_LONG = 'q',
  ULONG_LONG = 'Q',
  LONG_STORAGE = 'X',
  TENSOR = 'T',
  STORAGE = 'S',
  GENERATOR = 'G',
};

inline bool isFloat(RPCType t) {
  return (t == RPCType::FLOAT || t == RPCType::DOUBLE || t == RPCType::HALF);
}

inline bool isInteger(RPCType t) {
  return (t == RPCType::CHAR || t == RPCType::UCHAR ||
          t == RPCType::SHORT || t == RPCType:: USHORT ||
          t == RPCType::INT || t == RPCType::UINT ||
          t == RPCType::LONG || t == RPCType::ULONG ||
          t == RPCType::LONG_LONG || t == RPCType::ULONG_LONG);
}

inline const char* toString(RPCType t) {
  switch (t) {
    case RPCType::CHAR: return "Char";
    case RPCType::UCHAR: return "Byte";
    case RPCType::FLOAT: return "Float";
    case RPCType::DOUBLE: return "Double";
    case RPCType::HALF: return "Half";
    case RPCType::SHORT: return "Short";
    case RPCType::USHORT: return "UShort";
    case RPCType::INT: return "Int";
    case RPCType::UINT: return "UInt";
    case RPCType::LONG: return "Long";
    case RPCType::ULONG: return "ULong";
    case RPCType::LONG_LONG: return "LongLong";
    case RPCType::ULONG_LONG: return "ULongLong";
    case RPCType::LONG_STORAGE: return "LongStorage";
    case RPCType::TENSOR: return "Tensor";
    case RPCType::STORAGE: return "Storage";
    default: return "<unknown>";
  }
}

inline bool isObject(RPCType t) {
  return (t == RPCType::TENSOR || t == RPCType::STORAGE || t == RPCType::GENERATOR);
}

template<typename T>
struct type_traits {};

// NOTE: The `type` static constexpr variables of these specializations are
// additionally defined in RPCType.cpp to avoid undefined
// reference errors in C++11.
template<>
struct type_traits<char> {
  static constexpr RPCType type = RPCType::CHAR;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<int8_t> {
  static constexpr RPCType type = RPCType::CHAR;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<uint8_t> {
  static constexpr RPCType type = RPCType::UCHAR;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<float> {
  static constexpr RPCType type = RPCType::FLOAT;
  static constexpr bool is_floating_point = true;
};

template<>
struct type_traits<double> {
  static constexpr RPCType type = RPCType::DOUBLE;
  static constexpr bool is_floating_point = true;
};

template<>
struct type_traits<int16_t> {
  static constexpr RPCType type = RPCType::SHORT;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<uint16_t> {
  static constexpr RPCType type = RPCType::USHORT;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<int32_t> {
  static constexpr RPCType type = RPCType::INT;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<uint32_t> {
  static constexpr RPCType type = RPCType::UINT;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<int64_t> {
  static constexpr RPCType type = std::is_same<int64_t, long>::value ? RPCType::LONG : RPCType::LONG_LONG;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<uint64_t> {
  static constexpr RPCType type = std::is_same<uint64_t, unsigned long>::value ? RPCType::ULONG : RPCType::ULONG_LONG;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<std::conditional<std::is_same<int64_t, long>::value, long long, long>::type> {
  static constexpr RPCType type = std::is_same<int64_t, long>::value ? RPCType::LONG_LONG : RPCType::LONG;
  static constexpr bool is_floating_point = false;
};

template<>
struct type_traits<std::conditional<std::is_same<uint64_t, unsigned long>::value, unsigned long long, unsigned long>::type> {
  static constexpr RPCType type = std::is_same<uint64_t, unsigned long>::value ? RPCType::ULONG_LONG : RPCType::ULONG;
  static constexpr bool is_floating_point = false;
};

template<typename T>
struct type_traits<const T> : type_traits<T> {};

} // thd
